from cosypose.utils.tqdm import patch_tqdm; patch_tqdm()  # noqa
import torch.multiprocessing
import time
import json
import os
import glob
import sys

from collections import OrderedDict
import yaml
import argparse

import torch
import numpy as np
import pandas as pd
import pickle as pkl
import logging
from PIL import Image
from bokeh.io import export_png, save
import cosypose.visualization.plotter as plotter
import cosypose.visualization.singleview as sv_viewer
from cosypose.config import EXP_DIR, MEMORY, RESULTS_DIR, LOCAL_DATA_DIR
from cosypose.utils.timer import Timer
from cosypose.utils.distributed import init_distributed_mode, get_world_size

from cosypose.lib3d import Transform

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
from cosypose.evaluation.runner_utils import format_results, gather_predictions
from cosypose.utils.distributed import get_rank

from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset
from cosypose.datasets.bop import remap_bop_targets
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper
from cosypose.datasets.samplers import ListSampler

from cosypose.scripts.prediction_script import inference, load_detector, load_pose_models, crop_for_far, inference3, inference4, selectDetectorCoarseRefinerModel, filePath, camera_parametrization, renderImage, rgbgryToBool

def sequence(filename, data_path, rgb_path, object_set, camera_name, maximum = 2, renderBool = False, grayscale_bool = False,  nb_refine_it = 3):
    
    bbox_current_list=[]
    bbox_previous_list=[] 
    bbox_inter_list=[]
    
    # path initialization
    scene_path = data_path  / 'scene'
    
    # predictions parameters
    n_coarse_iterations = 1
    n_refiner_iterations = nb_refine_it

    # model handling  
    detector_run_id, coarse_run_id, refiner_run_id = selectDetectorCoarseRefinerModel(object_set)

    # camera parametrization
    camera, K = camera_parametrization(data_path, camera_name)

    # detector and predictor loading
    detector = load_detector(detector_run_id)
    predictor, mesh_db = load_pose_models(coarse_run_id, refiner_run_id, object_set=object_set)

    previousPose_init = False
    cropping = False
    number_images = len(os.listdir(scene_path))

    # lists for data storage
    frame_id = []
    image_name_list = []
    object_name = []
    pose = []
    bbox = []
    iter_duration = []
    status = []
    detection_score = []

    # clean and create the images folder
    dir_name = "img_yann/" + filename + '/'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    images = os.listdir(dir_name)

    for image in images:
        if image.endswith(".png") or image.endswith(".jpeg"):
            os.remove(os.path.join(dir_name, image))

    delta_t_predictions = []
    delta_t_detections = []
    delta_t_renderer = []
    delta_t_network = []

    for i in range (1,min(maximum+1, number_images)): 
        print('-'*81)
        print(i)
        print('-'*81)
        timer = Timer()
        timer.start()
        
        image_name = f'{str(i).zfill(4)}.png'
        rgb_path = scene_path / image_name

        # image formatting
        rgb = np.array(Image.open(rgb_path))
 
        if previousPose_init:
            detections, final_preds, all_preds, bbox_current_list, bbox_previous_list, bbox_inter_list= inference4(
                    detector, predictor, rgb, K,
                    TCO_init=TCO_init.cuda(),
                    n_coarse_iterations=0,
                    n_refiner_iterations=n_refiner_iterations)
        else:
            detections, final_preds, all_preds, delta_t = inference(
                    detector, predictor, rgb, K,
                    n_coarse_iterations=n_coarse_iterations,
                    n_refiner_iterations=n_refiner_iterations)

        # start with a good pose
        #if (i==1):
        #    pose_init = np.array([[ 0.9819119,  -0.18231227,  0.05110338,  0.00251109],
        #          [-0.04824508, -0.5019084,  -0.8635743,  -0.03091885],
        #          [ 0.1830891,   0.84548813, -0.5016254,   0.32253355],
        #          [ 0.,         0.,          0.,          1.        ]])
        #    pose_init = torch.tensor([pose_init]).float()
        #    tco_init = tc.PandasTensorCollection(infos=detections.infos, poses=pose_init)
        #    detections, final_preds,_ = inference(detector, predictor, rgb, K,
        #                TCO_init=tco_init.cuda(), n_coarse_iterations=0)

        if detections.infos.empty:
            previousPose_init = False
            frame_id.append(i)
            image_name_list.append(image_name)
            object_name.append(None)
            pose.append(None)
            bbox.append(None)
            detection_score.append(None)
            iter_duration.append(None)
            status.append("No detection")
            continue
        else:
            TCO_init = tc.PandasTensorCollection(infos=detections.infos, poses=final_preds.poses, bboxes=detections.bboxes)
            # Set to true if you want to warm start
            previousPose_init = True
            delta_t_detections.append(delta_t['detections'])
            delta_t_predictions.append(delta_t['predictions'])
            delta_t_renderer.append(delta_t['renderer'])
            delta_t_network.append(delta_t['network'])
            print(delta_t)

        #if len(detections.infos) != 1:
        #    previousPose_init = False

        
        # Saving all the predictions in the lists
        for idx, final_pose in enumerate(final_preds.poses.cpu()):
            frame_id.append(i)
            image_name_list.append(image_name)
            object_name.append(detections.infos.label[idx])
            pose.append(final_pose.cpu().numpy())
            bbox.append(detections.bboxes.cpu().numpy()[idx])
            detection_score.append(final_preds.cpu().infos.score[idx])
            iter_duration.append(delta_t)
            status.append("OK")


        # plotting
        if i < 9999:
            if renderBool:
                file = "img_yann/" + filename + '/'

                # plotIm = plt.plot_image(rgb)
                # figures = sv_viewer.make_singleview_custom_plot(rgb, camera,
                #     renderer, final_preds, detections)
                # export_png(figures['pred_overlay'], filename=f'images/{image_name}')
                renderImage(rgb_path, object_set, camera, final_preds, detections, file + f'result_{image_name}', grayscale_bool, bbox_current_list, bbox_previous_list, bbox_inter_list)

    # saving data
    df = pd.DataFrame(
            {
                "frame_id": frame_id,
                "image_name": image_name_list,
                "object_name": object_name,
                "pose": pose,
                "bbox": bbox,
                "detection_score":detection_score,
                "iter_duration": iter_duration,
                "status": status,
            }
        )

    df.to_pickle('results_' + filename + '.pkl', protocol=2)


    # time analysis
    print('Average time mask rcnn :' + str(np.mean(np.array(delta_t_detections))))
    print('Average time prediction :' + str(np.mean(np.array(delta_t_predictions))))
    print('Average time renderer :' + str(np.mean(np.array(delta_t_renderer))))
    print('Average time network :' + str(np.mean(np.array(delta_t_network))))

    return delta_t_detections, delta_t_detections, delta_t_renderer, delta_t_network

def main():

    nb_of_param = len(sys.argv)
    renderBool = False
    nb_of_imgs = 5
    grayscale_bool = False
    nb_refine_it = 3
    
    if (nb_of_param == 1):
        filename = "many_stairs"
        data_path, rgb_path, object_set, camera_name = filePath("many_stairs", 1)
    else:
        filename = sys.argv[1]
        data_path, rgb_path, object_set, camera_name = filePath(sys.argv[1], 1)
        if (nb_of_param >= 3):
            nb_of_imgs = int(sys.argv[2])
        if (nb_of_param >= 4):
            renderBool = eval(sys.argv[3])
        if (nb_of_param >= 5):
            try:
                grayscale_bool = eval(sys.argv[4])
            except NameError:
                grayscale_bool = rgbgryToBool(sys.argv[4])
        if (nb_of_param >= 6):
            nb_refine_it = int(sys.argv[5])

    sequence(filename, data_path, rgb_path, object_set, camera_name, nb_of_imgs, renderBool, grayscale_bool, nb_refine_it)
    
if __name__ == '__main__':
    main()
