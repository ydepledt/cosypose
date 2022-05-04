from cosypose.utils.tqdm import patch_tqdm; patch_tqdm()  # noqa
import torch.multiprocessing
import time
import glob
import sys
from os.path import basename, splitext
import json

from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
import yaml
import argparse

import torch
import cv2
import numpy as np
import pandas as pd
import pickle as pkl
import logging
from PIL import Image, ImageOps
from bokeh.io import export_png, save

import cosypose.visualization.plotter as plotter
import cosypose.visualization.singleview as sv_viewer
from cosypose.config import EXP_DIR, MEMORY, RESULTS_DIR, LOCAL_DATA_DIR

from cosypose.utils.distributed import init_distributed_mode, get_world_size

from cosypose.lib3d import Transform
from cosypose.lib3d.camera_geometry import get_K_crop_resize
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import check_update_config as check_update_config_pose
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse
from cosypose.training.detector_models_cfg import check_update_config as check_update_config_detector
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.integrated.detector import Detector
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor

from cosypose.evaluation.meters.pose_meters import PoseErrorMeter
from cosypose.evaluation.pred_runner.multiview_predictions import MultiviewPredictionRunner
from cosypose.evaluation.eval_runner.pose_eval import PoseEvaluation

import cosypose.utils.tensor_collection as tc
from cosypose.utils.timer import Timer
from cosypose.evaluation.runner_utils import format_results, gather_predictions
from cosypose.utils.distributed import get_rank


from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset
from cosypose.datasets.bop import remap_bop_targets
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper
from cosypose.datasets.samplers import ListSampler


def load_detector(run_id):
    """Load the detector"""

    run_dir = EXP_DIR / run_id  #path of model
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt) #parameters and buffers loaded
    model = model.cuda().eval() #send model to current device
    model.cfg = cfg 
    model.config = cfg
    model = Detector(model)
    return model

def load_pose_models(coarse_run_id, refiner_run_id=None, n_workers=8, object_set='tless'):
    """Load the predictor"""

    run_dir = EXP_DIR / coarse_run_id

    if object_set == 'tless':
        object_ds_name, urdf_ds_name = 'tless.bop', 'tless.cad'
    elif object_set == 'ycbv_stairs':
        object_ds_name, urdf_ds_name = 'ycbv_stairs', 'ycbv_stairs'
    else:
        object_ds_name, urdf_ds_name = 'ycbv.bop-compat.eval', 'ycbv'    

    object_ds = make_object_dataset(object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=urdf_ds_name, n_workers=n_workers, preload_cache=False)

    mesh_db_batched = mesh_db.batched().cuda()
    def load_model(run_id):
        if run_id is None:
            return
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_pose(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        return model

    coarse_model = load_model(coarse_run_id)
    refiner_model = load_model(refiner_run_id)
    model = CoarseRefinePosePredictor(coarse_model=coarse_model,
        refiner_model=refiner_model) #Create a subclass of nn.Module (neural network)
    return model, mesh_db

def crop_for_far(rgb, K):
    """
    Experiment...
    """
    h,w = rgb.shape[:2]
    new_h, new_w = int(h/2), int(w/2)

    new_im_1 = rgb[0:new_h, 0:new_w, :]
    new_im_2 = rgb[new_h:, new_w:, :]
    new_im_3 = rgb[0:new_h, new_w:, :]
    new_im_4 = rgb[new_h:, 0:new_w, :]
    imgs = [new_im_1, new_im_2, new_im_3, new_im_4]

    K_init = torch.zeros(4,3,3)
    for i in range(4):
        K_init[i,:,:] = K

    boxes = torch.tensor([[0, 0, new_w, new_h],
        [new_w, new_h, w, h],
        [new_w, 0, w, new_h],
        [0, new_h, new_w, h]])
                                                        
    K = get_K_crop_resize(K_init, boxes, (h,w), (new_h,new_w))
    return imgs, K

def image_formating(rgb):
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    rgb = rgb[..., :3]
    h, w = rgb.shape[:2]
    rgb = torch.as_tensor(rgb)
    rgb = rgb.unsqueeze(0)
    rgb = rgb.cuda().float().permute(0, 3, 1, 2) / 255
    return rgb


def inference(detector, predictor, rgb, K, TCO_init=None, n_coarse_iterations=1, n_refiner_iterations=2):
    """
    A dumy function that initialises with TCO_init, may crash if the detections doesn't match the init
    """

    # C_p = TCO * O_p  transformation entre repère caméra et objet

    # image formatting
    rgb = image_formating(rgb)
    
    # prediction running
    timer_detection = Timer()
    timer_detection.start()
    detections = detector.get_detections(images=rgb, 
            one_instance_per_class=False,
            detection_th = 0.95,
            output_masks = None,
            mask_th = 0.1)
    print("\n", detections, "\n")
    delta_t_detections = timer_detection.stop()
    
    if detections.infos.empty:
        print('-'*80)
        print('No object detected')
        print('-'*80)
        return detections, None, None, None
    else:
        timer_prediction = Timer()
        timer_prediction.start()
        final_preds, all_preds = predictor.get_predictions(
                rgb, K, detections=detections,
                data_TCO_init = TCO_init,
                n_coarse_iterations=n_coarse_iterations,
                n_refiner_iterations=n_refiner_iterations,
            )
        
        delta_t_prediction = timer_prediction.stop()
        delta_t_render = all_preds['delta_t_render']
        delta_t_net = all_preds['delta_t_net']
        
        delta_t = {'detections':delta_t_detections.total_seconds(),
                'predictions':delta_t_prediction.total_seconds(),
                'renderer':np.mean(np.array(delta_t_render)),
                'network':np.mean(np.array(delta_t_net))}
                
        return detections, final_preds, all_preds, delta_t

def bbox_center(bbox):
    u = (bbox[0]+bbox[2])/2
    v = (bbox[1]+bbox[3])/2
    large = np.sqrt((bbox[2]-bbox[0])*(bbox[2]-bbox[0]) +
            (bbox[3]-bbox[1])*(bbox[3]-bbox[1]))/2
    return u,v,large


def inference2(detector, predictor, rgb, K, poses_init=None, n_coarse_iterations=1, n_refiner_iterations=2):
    """
    This one is initializing each object that is detected and available in poses_init

    BUT: - do a forward pass per object: very time inefficient for scenes with several objects
    - assume that there is one object per class
    """
    
    # image formatting
    rgb = image_formating(rgb)

    # prediction running
    detections = detector.get_detections(images=rgb,
            one_instance_per_class=False,
            detection_th = 0.99,
            output_masks = None,
            mask_th = 0.5)
            
    if detections.infos.empty:
        print('-'*80)
        print('No object detected')
        print('-'*80)
        return detections, None, None
    else:
        TCO_init_object = None
        counter = 0
        # We go through each detection to initialize if necessary
        for detection in detections:
            infos = dict(score=[detection.infos[2]], label=[detection.infos[1]], batch_im_id=[detection.infos[0]])
            detection = tc.PandasTensorCollection(
                    infos=pd.DataFrame(infos),
                    bboxes=detection.bboxes.unsqueeze(0))
            bb_infos = bbox_center(detection.bboxes.cpu().numpy()[0])
            if not (poses_init is None):
                # We check if the object is in the init df
                TCO_infos = poses_init.infos.loc[poses_init.infos == detection.infos['label'][0]]
                TCO_infos = poses_init.infos
                TCO_init_object = None
                if not TCO_infos.empty:
                    min_dist_bb = bb_infos[2] 
                    for k in range(len(TCO_infos)):
                        if (k==0)or(k==4)or(k==5):
                            continue
                        bbox_wolf = poses_init.object_coords[TCO_infos.index.values[k]]
                        dist_bb = np.linalg.norm(bbox_wolf.cpu().numpy() - np.array([bb_infos[0], bb_infos[1]]))
                        #print('bbox distance : ' + str(dist_bb))
                        if (dist_bb < min_dist_bb):
                            print('init with dist_bb ' + str(dist_bb))
                            min_dist_bb = dist_bb
                            TCO_pose = poses_init.poses[TCO_infos.index.values[k]]
                            TCO_init_object = tc.PandasTensorCollection(
                                infos=detection.infos.iloc[[0]],
                                poses=TCO_pose.unsqueeze(0))
                else:
                    TCO_init_object = None

            if TCO_init_object is None:
                final_preds_object, all_preds = predictor.get_predictions(
                        rgb, K, detections=detection,
                        data_TCO_init = TCO_init_object,
                        n_coarse_iterations=1,
                        n_refiner_iterations=n_refiner_iterations,
                    )
            else:
                final_preds_object, all_preds = predictor.get_predictions(
                        rgb, K, detections=detection,
                        data_TCO_init = TCO_init_object,
                        n_coarse_iterations=0,
                        n_refiner_iterations=n_refiner_iterations,
                    )

            # A trick to concatenate the results
            if counter == 0:
                final_preds = final_preds_object
                final_preds_info = final_preds.infos
            else:
                final_preds.poses = torch.cat((final_preds.poses, final_preds_object.poses))
                final_preds_info = pd.concat([final_preds_info, final_preds_object.infos], ignore_index = True)
                final_preds.infos = final_preds_info
            counter += 1
        return detections, final_preds, all_preds


def inference3(detector, predictor, rgb, K, TCO_init=None, n_coarse_iterations=1, n_refiner_iterations=2):
    """
    This one is initializing each object that is detected and available in poses_init

    BUT: - do a forward pass per object: very time inefficient for scenes with several objects
    - assume that there is one object per class
    """
    
    # image formatting
    rgb = image_formating(rgb)

    # prediction running
    detections = detector.get_detections(images=rgb,
            one_instance_per_class=False,
            detection_th = 0.98,
            output_masks = None,
            mask_th = 0.5)
    if detections.infos.empty:
        print('-'*80)
        print('No object detected')
        print('-'*80)
        return detections, None, None
    else:
        TCO_init_object = None
        counter = 0
        # We go through each detection to initialize if necessary
        for detection in detections:
            label = detection.infos[1]
            infos = dict(score=[detection.infos[2]], label=[detection.infos[1]], batch_im_id=[detection.infos[0]])
            detection = tc.PandasTensorCollection(
                    infos=pd.DataFrame(infos),
                    bboxes=detection.bboxes.unsqueeze(0))
            bb_infos = bbox_center(detection.bboxes.cpu().numpy()[0])
            if not (TCO_init is None):
                # We check if the object is in the init df
                #TCO_infos = TCO_init.infos.loc[TCO_init['label'] == detection.infos['label'][0]]
                TCO_init_object = None
    
                for k in range(len(TCO_init.infos)):
                    if (TCO_init.infos['label'][k] == label):
                        bbox_det = bbox_center(TCO_init[k].bboxes.cpu().numpy())
                        dist_bb = np.linalg.norm(np.array([bbox_det[0], bbox_det[1]]) - np.array([bb_infos[0], bb_infos[1]])) #np.linalg.norm(np.array([bbox_det[0], bbox_det[1]]) - np.array([bb_infos[0], bb_infos[1]]))
                        print('bbox distance : ' + str(dist_bb))

                        if (dist_bb < bb_infos[2]):
                            TCO_pose = TCO_init.poses[k]
                            TCO_init_object = tc.PandasTensorCollection(
                                infos=detection.infos.iloc[[0]],
                                poses=TCO_pose.unsqueeze(0))
                    

            if TCO_init_object is None:
                final_preds_object, all_preds = predictor.get_predictions(
                        rgb, K, detections=detection,
                        data_TCO_init = TCO_init_object,
                        n_coarse_iterations=1,
                        n_refiner_iterations=n_refiner_iterations,
                    )
            else:
                final_preds_object, all_preds = predictor.get_predictions(
                        rgb, K, detections=detection,
                        data_TCO_init = TCO_init_object,
                        n_coarse_iterations=0,
                        n_refiner_iterations=n_refiner_iterations,
                    )

            # A trick to concatenate the results
            if counter == 0:
                final_preds = final_preds_object
                final_preds_info = final_preds.infos
            else:
                final_preds.poses = torch.cat((final_preds.poses, final_preds_object.poses))
                final_preds_info = pd.concat([final_preds_info, final_preds_object.infos], ignore_index = True)
                final_preds.infos = final_preds_info
            counter += 1
        print(final_preds.poses)
        return detections, final_preds, all_preds

def filePath(object, imgNb = 1):
    data_path = LOCAL_DATA_DIR / object
    rgb_path = data_path / 'scene' /  (str(imgNb).zfill(4) + ".png")
    filepath = glob.glob(str(data_path) + '/*.json')[0]
    camera_name = splitext(basename(filepath))[0] + ".json"
    
    if ((object.lower().find("stair") != -1) or (object.lower().find("ycbv_stairs") != -1)):
        object_set = "ycbv_stairs"
    elif ((object.lower().find("soup") != -1) or ((object.lower().find("ycbv") != -1) and (object.lower().find("ycbv_stairs") == -1))):
        object_set = "ycbv"
    elif ((object.lower().find("switch") != -1) or (object.lower().find("powerstrip") != -1) or (object.lower().find("tless") != -1)):
        object_set = "tless"
    else:
        object_set = "tless"
    
    return data_path, rgb_path, object_set, camera_name

def IOU(box1, box2):
    x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
    x3, y3, x4, y4 = box2[0], box2[1], box2[2], box2[3] 
    x_inter1 = max(x1,x3)
    y_inter1 = max(y1,y3)
    x_inter2 = min(x2,x4)
    y_inter2 = min(y2,y4)
    box_inter = [x_inter1, y_inter1, x_inter2, y_inter2]
    width_inter = abs(x_inter2 - x_inter1)
    height_inter = abs(y_inter2 - y_inter1)
    area_inter = width_inter * height_inter
    width_box1 = abs(x2-x1)
    height_box1 = abs(y2-y1)
    width_box2 = abs(x4-x3)
    height_box2 = abs(y4-y3)
    area_box1 = width_box1 * height_box1    
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter/area_union
    return iou, box_inter

def argmax(list):
    max = 0
    index = -1
    for i in range(len(list)):
        if max < list[i] and list[i] > 0 and list[i] < 1:
            max = list[i]
            index = i
    return index


def inference4(detector, predictor, rgb, K, TCO_init=None, n_coarse_iterations=1, n_refiner_iterations=2):
    """
    This one is initializing each object that is detected and available in poses_init

    BUT: - do a forward pass per object: very time inefficient for scenes with several objects
    """

    bbox_inter_list    = []
    bbox_current_list  = []
    bbox_previous_list = []
    iou = 0
    
    # image formatting
    rgb = image_formating(rgb)

    # prediction running
    detections = detector.get_detections(images=rgb,
            one_instance_per_class=False,
            detection_th = 0.98,
            output_masks = None,
            mask_th = 0.5)
    if detections.infos.empty:
        print('-'*80)
        print('No object detected')
        print('-'*80)
        return detections, None, None
    else:
        TCO_init_object = None
        counter = 0
        # We go through each detection to initialize if necessary
        for detection in detections:
            label = detection.infos[1]
            infos = dict(score=[detection.infos[2]], label=[detection.infos[1]], batch_im_id=[detection.infos[0]])
            detection = tc.PandasTensorCollection(
                    infos=pd.DataFrame(infos),
                    bboxes=detection.bboxes.unsqueeze(0))
            bb_infos = detection.bboxes.cpu().numpy()[0]
        
            bbox_current_list.append(bb_infos)
        
            if not (TCO_init is None):
                # We check if the object is in the init df
                #TCO_infos = TCO_init.infos.loc[TCO_init['label'] == detection.infos['label'][0]]
                TCO_init_object = None
                
                print(len(TCO_init.infos['label']))
                bbox_dets   = []
                bbox_inters = []
                ious        = []
                k_list      = []
                for k in range(len(TCO_init.infos)):
                    if (TCO_init.infos['label'][k] == label):
                        
                        bbox_det = TCO_init[k].bboxes.cpu().numpy()
                        iou, bbox_inter = IOU(bb_infos, bbox_det)
                        bbox_inter.append(iou)

                        bbox_dets.append(bbox_det)
                        bbox_inters.append(bbox_inter)
                        ious.append(iou)
                        k_list.append(k)

                #Give the index of the best pose init
                index = -1
                if len(ious) != 0:
                    index = argmax(ious)
                
                if index != -1:
                    bbox_previous_list.append(bbox_dets[index]) 
                    bbox_inter_list.append(bbox_inters[index])

                    if (ious[index] > 0.80):
                        TCO_pose = TCO_init.poses[k_list[index]]
                        TCO_init_object = tc.PandasTensorCollection(
                            infos=detection.infos.iloc[[0]],
                            poses=TCO_pose.unsqueeze(0))
                    

            if TCO_init_object is None:
                final_preds_object, all_preds = predictor.get_predictions(
                        rgb, K, detections=detection,
                        data_TCO_init = TCO_init_object,
                        n_coarse_iterations=1,
                        n_refiner_iterations=n_refiner_iterations,
                    )
            else:
                final_preds_object, all_preds = predictor.get_predictions(
                        rgb, K, detections=detection,
                        data_TCO_init = TCO_init_object,
                        n_coarse_iterations=0,
                        n_refiner_iterations=n_refiner_iterations,
                    )

            # A trick to concatenate the results
            if counter == 0:
                final_preds = final_preds_object
                final_preds_info = final_preds.infos
            else:
                final_preds.poses = torch.cat((final_preds.poses, final_preds_object.poses))
                final_preds_info = pd.concat([final_preds_info, final_preds_object.infos], ignore_index = True)
                final_preds.infos = final_preds_info
            counter += 1
        print(final_preds.poses)
        return detections, final_preds, all_preds, bbox_current_list, bbox_previous_list, bbox_inter_list

def selectDetectorCoarseRefinerModel(object_set):
    if object_set == 'tless':
        detector_run_id = 'detector-bop-tless-synt+real--452847'
        coarse_run_id = 'coarse-bop-tless-synt+real--160982'
        refiner_run_id = 'refiner-bop-tless-synt+real--881314'
    elif object_set == 'ycbv_stairs':
        detector_run_id = 'detector-ycbv_stairs--720976'
        coarse_run_id = 'ycbv_stairs-coarse-4GPU-fixed-434372'
        refiner_run_id = 'ycbv_stairs-refiner-4GPU-4863'
    elif object_set == 'ycbv':
        detector_run_id = 'detector-bop-ycbv-synt+real--292971'
        coarse_run_id = 'coarse-bop-ycbv-synt+real--822463'
        refiner_run_id = 'refiner-bop-ycbv-synt+real--631598'
    return detector_run_id, coarse_run_id, refiner_run_id

def drawDetections(image, bbox_current_list, bbox_previous_list, bbox_inter_list):
    for bbox_current in bbox_current_list:
        image = cv2.rectangle(image, (bbox_current[0],bbox_current[1]), (bbox_current[2],bbox_current[3]), (255,0,0), 2)
    for bbox_previous in bbox_previous_list:
        image = cv2.rectangle(image, (bbox_previous[0],bbox_previous[1]), (bbox_previous[2],bbox_previous[3]), (0,255,0), 2)
    for bbox_inter in bbox_inter_list:
        image = cv2.rectangle(image, (bbox_inter[0],bbox_inter[1]), (bbox_inter[2],bbox_inter[3]), (0,0,255), 1)
        right_top_corner = (int(bbox_inter[2]) + 10, int(bbox_inter[1]) + 10)
        print(right_top_corner)
        image = cv2.putText(image, "Iou = " + str(round(bbox_inter[4], 4)), right_top_corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)


def renderImage(rgb_path, object_set, camera, final_preds, detections, name, grayscale_bool, bbox_current_list=[], bbox_previous_list=[], bbox_inter_list=[]):

    # load image for render
    rgb = Image.open(rgb_path)
    if (grayscale_bool):
        grayscale_im = np.array(ImageOps.grayscale(rgb))
        image = np.zeros((grayscale_im.shape[0],grayscale_im.shape[1],3), dtype=int)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i,j] = grayscale_im[i,j]
        image = image.astype(np.uint8)
    else:
        image = np.array(rgb).astype(np.uint8)
    
    # plotting
    plt = plotter.Plotter()
    if object_set == 'tless':
        obj_ds_name = 'tless.cad'
    elif object_set == 'ycbv_stairs':
        obj_ds_name = 'ycbv_stairs'
    else:
        obj_ds_name = 'ycbv'
    renderer = BulletSceneRenderer(obj_ds_name, gpu_renderer=True)

    # render
    plotIm = plt.plot_image(image)
    figures = sv_viewer.make_singleview_custom_plot(image, camera, renderer, final_preds, detections)

    # figures['pred_overlay'] = np.array(figures['pred_overlay'])

    drawDetections(image, bbox_current_list, bbox_previous_list, bbox_inter_list)

    export_png(figures['pred_overlay'], filename=f'{name}')
   

    grayscale_im = Image.fromarray(image)
    
    if sys.argv[0][27:-10] == "prediction":
        grayscale_im.save("input.jpeg")
    elif sys.argv[0][27:-10] == "sequence":
        grayscale_im.save(name[:-15] + "input_" + name[-8:-4] + ".jpeg") #input_XXXX.jpeg


def camera_parametrization(data_path, camera_name):
    cam_infos = json.loads((data_path / camera_name).read_text())
    K_ = np.array([[cam_infos['fx'], 0.0, cam_infos['cx']],
            [0.0, cam_infos['fy'], cam_infos['cy']],
            [0.0, 0.0, 1]])
    K = torch.as_tensor(K_)
    K = K.unsqueeze(0)
    K = K.cuda().float()
    TC0 = Transform(np.eye(3), np.zeros(3))
    T0C = TC0.inverse()
    T0C = T0C.toHomogeneousMatrix()
    h = cam_infos['height']
    w = cam_infos['width']
    return dict(T0C=T0C, K=K_, TWC=T0C, resolution=(h,w)), K

def rgbgryToBool(color):
    if "RGB" == color or "COLOR" == color or 'C' == color:
        return False
    elif "GREY" == color or "GRY" == color or 'G' == color:
        return True
    else:
        print("Error in color selection, putting grayscale as default...")
        return True

def main():
    # path initialization
    nb_of_param = len(sys.argv)
    renderBool = False
    n_refiner_iterations = 3
    grayscale_bool = False
    
    if (nb_of_param < 3):
        if (nb_of_param == 1):
            data_path, rgb_path, object_set, camera_name = filePath("many_stairs", 1)
        if (nb_of_param == 2):
            data_path, rgb_path, object_set, camera_name = filePath(sys.argv[1], 1)
    else:
        data_path, rgb_path, object_set, camera_name = filePath(sys.argv[1], int(sys.argv[2]))
        if (nb_of_param >= 4):
            renderBool = eval(sys.argv[3])
        if (nb_of_param >= 5):
            try:
                grayscale_bool = eval(sys.argv[4])
            except NameError:
                grayscale_bool = rgbgryToBool(sys.argv[4])
        if (nb_of_param >= 6):
            n_refiner_iterations = int(sys.argv[5])
            
    # model handling
    detector_run_id, coarse_run_id, refiner_run_id = selectDetectorCoarseRefinerModel(object_set)

    # camera parametrization

    camera, K = camera_parametrization(data_path, camera_name)

    # detector and predictor loading
    detector = load_detector(detector_run_id)
    predictor, mesh_db = load_pose_models(coarse_run_id, refiner_run_id, object_set=object_set)
    
    rgb = Image.open(rgb_path)
    rgb = np.array(rgb)

    detections, final_preds,_,_ = inference(detector, predictor, rgb, K, n_coarse_iterations=1, n_refiner_iterations=n_refiner_iterations)
   
    print(final_preds.poses)

    if (renderBool):
        renderImage(rgb_path, object_set, camera, final_preds, detections, "result.png", grayscale_bool)

    
if __name__ == '__main__':
    main()
