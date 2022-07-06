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

import os
os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.2'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '420'


def load_detector(run_id):
    """Gets and returns the detector, given an ID

    Args:
        run_id (string) : represent the ID of the detector (ex : 'detector-bop-tless-synt+real--452847')

    Returns:
        model  (Detector object)
    """

    run_dir = EXP_DIR / run_id  # path of the model (local_data/experiments/model's ID)

    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader) # load config file
    cfg = check_update_config_detector(cfg)

    label_to_category_id = cfg.label_to_category_id

    model = create_model_detector(cfg, len(label_to_category_id)) # create a mask for detector (DetectorMaskRCNN object)
    
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt) #parameters and buffers loaded
    model = model.cuda().eval() #send model to current device
    model.cfg = cfg 
    model.config = cfg

    model = Detector(model) # create a Detector object

    return model

def load_pose_models(coarse_run_id, refiner_run_id=None, n_workers=8, object_set='tless'):
    """Gets and returns predictors (coarse and refiner), given an ID

    Args:
        coarse_run_id  (string) : ID of the coarse  predictor (ex : 'coarse-bop-tless-synt+real--160982')
        refiner_run_id (string) : ID of the refiner predictor (ex : 'ycbv_stairs-refiner-4GPU-4863')
        n_workers      (int)    : number of workers for the renderer
        object_set     (string) : type of dataset (ex : 'tless', 'ycbv')    
        

    Returns:
        model          (Detector object)
        mesh_db        (MeshDataBase object)
    """

    run_dir = EXP_DIR / coarse_run_id # path of the coarse model (local_data/experiments/model's ID)

    if object_set == 'tless': 
        object_ds_name, urdf_ds_name = 'tless.bop', 'tless.cad'
    elif object_set == 'ycbv_stairs':
        object_ds_name, urdf_ds_name = 'ycbv_stairs', 'ycbv_stairs'
    else:
        object_ds_name, urdf_ds_name = 'ycbv.bop-compat.eval', 'ycbv'    

    object_ds = make_object_dataset(object_ds_name) # create BOPObjectDataset object

    mesh_db         = MeshDataBase.from_object_ds(object_ds) # create a mesh of the object (MeshDataBase object)
    mesh_db_batched = mesh_db.batched().cuda()

    renderer = BulletBatchRenderer(object_set=urdf_ds_name, n_workers=n_workers, preload_cache=False) # create BulletBatchRenderer object for future rendering


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


    coarse_model  = load_model(coarse_run_id)  # coarse loaded
    refiner_model = load_model(refiner_run_id) # refiner loaded

    model = CoarseRefinePosePredictor(coarse_model=coarse_model,
        refiner_model=refiner_model) # create a subclass of nn.Module (neural network)

    return model, mesh_db

def sceneInformation(dataset_name, image_idx = 1):

    """Gets and returns all informations about the scene and the image

    Args:
        dataset_name (string) : name of the dataset (ex : soup1, powerstrip1)
        image_idx    (int)    : index of the image in the scene     
        

    Returns:
        dataset_path (string) : path to dataset (local_data/dataset_name)
        image_path   (string) : path to image   (local_data/dataset_name/image_idx.png)
        object_set   (string) : type of dataset (ex : 'tless', 'ycbv') 
        camera_name  (string) : name of the camera + .json
    """

    dataset_path = LOCAL_DATA_DIR / dataset_name

    image_path = dataset_path / 'scene' /  (str(image_idx).zfill(4) + ".png")

    filepath = glob.glob(str(dataset_path) + '/*.json')[0]
    camera_name = splitext(basename(filepath))[0] + ".json"
    
    # check if dataset is based on stairs model
    if ((dataset_name.lower().find("stair") != -1) or (dataset_name.lower().find("ycbv_stairs") != -1)):
        object_set = "ycbv_stairs" 
    
    # check if dataset is based on soup can model
    elif ((dataset_name.lower().find("soup") != -1) or ((dataset_name.lower().find("ycbv") != -1) and (dataset_name.lower().find("ycbv_stairs") == -1))):
        object_set = "ycbv"

    # check if dataset is based on switches or powerstrips model
    elif ((dataset_name.lower().find("switch") != -1) or (dataset_name.lower().find("powerstrip") != -1) or (dataset_name.lower().find("tless") != -1)):
        object_set = "tless"

    else:
        object_set = "tless"
    
    return dataset_path, image_path, object_set, camera_name

def IOU(box1, box2):

    """Gets and returns the area of intersection of two boxes

    Args:
        box1      (list[double]) : list of box's corners coordinate
        box2      (list[double]) : list of box's corners coordinate     

        box1[0],box[1]____________________
               |                          |                          
               |                          |
               |                          |
               |                          |
               |___________________box1[2],box[3]
                

    Returns:
        iou       (double)       : area of boxes' intersection over area of boxes' union
        box_inter (list[double]) : list of intersection box's corners  
    """

    x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3] # box 1 corners
    x3, y3, x4, y4 = box2[0], box2[1], box2[2], box2[3] # box 2 corners

    x_inter1 = max(x1,x3)
    y_inter1 = max(y1,y3)
    x_inter2 = min(x2,x4)
    y_inter2 = min(y2,y4)

    box_inter = [x_inter1, y_inter1, x_inter2, y_inter2] # intersection of the two boxes
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

    """Gets and returns max's index 

    Args:
        list  (list[])      


    Returns:
        index (int) : index of the max (greater than 0 and smaller than 1) 
    """
    max = 0
    index = -1

    for i in range(len(list)):
        if max < list[i] and list[i] > 0 and list[i] < 1:
            max = list[i]
            index = i

    return index

def bbox_center(bbox):

    """Gets and returns bbox center

    Args:
        bbox  (list[double]) : list of bbox's corners coordinate   


    Returns:
        u     (double)       : xcoordinates of center
        v     (double)       : ycoordinates of center
        large (double)       : lenght of diagonal's half
    """

    u = (bbox[0]+bbox[2])/2
    v = (bbox[1]+bbox[3])/2

    large = np.sqrt((bbox[2]-bbox[0])**2 +(bbox[3]-bbox[1])**2)/2

    return u,v,large 

def image_formating(image):

    """Formates an image given and returns it

    Args:
        image (np.array) : image in color   


    Returns:
        image (np.array) : image formated in color
    """

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)

    image = image[..., :3]
    image = torch.as_tensor(image)
    image = image.unsqueeze(0)
    image = image.cuda().float().permute(0, 3, 1, 2) / 255

    return image

def renderImage(image_path, object_set, camera, final_preds, detections, rendered_img_name, grayscale_img, bbox_current_list=[], bbox_previous_list=[], bbox_inter_list=[]):

    """Create both input and result png image

    Args:
        image_path         (string)                        : path to image   (local_data/dataset_name/image_idx.png)
        object_set         (string)                        : type of dataset (ex : 'tless', 'ycbv')
        camera             (matrix)                        
        final_preds        (dict)                          : last predictions from predictor
        detections         (PandasTensorCollection object) : detections from detector
        rendered_img_name  (string)                        : name of the png result
        grayscale_img      (bool)                          : result in grayscale or color
        bbox_current_list  (list[list[double]])            : list of current box's corners coordinate for each object in the scene (ex: [[x1,y1,x2,y2], [x1_2,y1_2,x2_2,y2_2], ...])                
        bbox_previous_list (list[list[double]])            : list of previous box's corners coordinate for each object in the scene                
        bbox_inter_list    (list[list[double]])            : list of intersection box's corners coordinate for each object in the scene                 

    """

    # load image for render
    image = Image.open(image_path)

    # if user want a result in grayscale
    if (grayscale_img):
        grayscale_image = np.array(ImageOps.grayscale(image)) # convert image to grayscale
        image = np.zeros((grayscale_image.shape[0],grayscale_image.shape[1],3), dtype=int)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i,j] = grayscale_image[i,j]

        image = image.astype(np.uint8)

    else:
        image = np.array(image).astype(np.uint8)
    
    # plotting
    plt = plotter.Plotter()
    if object_set == 'tless':
        obj_ds_name = 'tless.cad'
    elif object_set == 'ycbv_stairs':
        obj_ds_name = 'ycbv_stairs'
    else:
        obj_ds_name = 'ycbv'
    renderer = BulletSceneRenderer(obj_ds_name, gpu_renderer=True)

    # render the image
    figures = sv_viewer.make_singleview_custom_plot(image, camera, renderer, final_preds, detections)

    # draw bounding_boxex detections for all detected objects in input image
    drawDetections(image, bbox_current_list, bbox_previous_list, bbox_inter_list)

    # create a png image with poses as overlay
    export_png(figures['pred_overlay'], filename=f'{rendered_img_name}')
   
    image = Image.fromarray(image)
    
    # name of images result depend on the script launched
    if sys.argv[0][27:-10] == "prediction":
        image.save("input.jpeg") # only one result with prediction
    elif sys.argv[0][27:-10] == "sequence":
        image.save(rendered_img_name[:-15] + "input_" + rendered_img_name[-8:-4] + ".jpeg") # form of images results : input_XXXX.jpeg


def camera_parametrization(dataset_path, camera_name):

    """Gets path to camera and returns dict with camera transformation

    Args:
        dataset_path (string)       : path to dataset (local_data/dataset_name)
        camera_name  (string)       : name of the camera + .json

    Returns:
        camera_info  (dict)         : transformations and height/width of the camera
        K            (torch.tensor) : intrinsics matrix

    """

    # load a dict based on camera.json file with all cam caracteristics
    cam_infos = json.loads((dataset_path / camera_name).read_text())

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

    camera_info = dict(T0C=T0C, K=K_, TWC=T0C, resolution=(h,w))

    return camera_info, K


def rgbgryToBool(color):

    """Gets a color and returns a bool 

    Args:
        color (string) : color user wants

    Returns:
              (bool)   : Grayscale->True, Colored->False

    """

    if "RGB" == color.upper() or "COLOR" == color.upper() or 'C' == color.upper() or 'COL' == color.upper():
        return False
    elif 'GRAYSCALE' == color.upper() or "GREY" == color.upper() or "GRY" == color.upper() or 'G' == color.upper():
        return True
    else:
        print("Error in color selection, putting grayscale as default...")
        return True


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

def main():
    # path initialization
    nb_of_param = len(sys.argv)
    renderBool = False
    n_refiner_iterations = 3
    grayscale_img = False
    
    if (nb_of_param < 3):
        if (nb_of_param == 1):
            dataset_path, image_path, object_set, camera_name = sceneInformation("many_stairs", 1)
        if (nb_of_param == 2):
            dataset_path, image_path, object_set, camera_name = sceneInformation(sys.argv[1], 1)
    else:
        dataset_path, image_path, object_set, camera_name = sceneInformation(sys.argv[1], int(sys.argv[2]))
        if (nb_of_param >= 4):
            renderBool = eval(sys.argv[3])
        if (nb_of_param >= 5):
            try:
                grayscale_img = eval(sys.argv[4])
            except NameError:
                grayscale_img = rgbgryToBool(sys.argv[4])
        if (nb_of_param >= 6):
            n_refiner_iterations = int(sys.argv[5])
            
    # model handling
    detector_run_id, coarse_run_id, refiner_run_id = selectDetectorCoarseRefinerModel(object_set)

    # camera parametrization

    camera, K = camera_parametrization(dataset_path, camera_name)

    # detector and predictor loading
    detector = load_detector(detector_run_id)
    predictor, mesh_db = load_pose_models(coarse_run_id, refiner_run_id, object_set=object_set)
    
    rgb = Image.open(image_path)
    rgb = np.array(rgb)

    detections, final_preds,_,_ = inference(detector, predictor, rgb, K, n_coarse_iterations=1, n_refiner_iterations=n_refiner_iterations)
   
    print(final_preds.poses)

    if (renderBool):
        renderImage(image_path, object_set, camera, final_preds, detections, "result.png", grayscale_img)

    
if __name__ == '__main__':
    main()







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
