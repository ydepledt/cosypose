from cosypose.utils.tqdm import patch_tqdm; patch_tqdm()  # noqa
import torch.multiprocessing
import time
import json
import os

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

from cosypose.scripts.prediction_script import inference, load_detector, load_pose_models, crop_for_far, inference2

def main():
    # path initialization
    data_path = LOCAL_DATA_DIR / 'solo_stairs3' 
    scene_path = data_path  / 'scene'
    init_path = data_path / 'init.csv'
    
    # predictions parameters
    n_coarse_iterations = 1
    n_refiner_iterations = 8

    # model handling
    object_set = 'ycbv_stairs'
    
    if object_set == 'tless':
        detector_run_id = 'detector-bop-tless-synt+real--452847'
        coarse_run_id = 'coarse-bop-tless-synt+real--160982'
        refiner_run_id = 'refiner-bop-tless-synt+real--881314'
    elif object_set == 'ycbv':
        detector_run_id = 'detector-bop-ycbv-synt+real--292971'
        coarse_run_id = 'coarse-bop-ycbv-synt+real--822463'
        refiner_run_id = 'refiner-bop-ycbv-synt+real--631598'
    else:
        detector_run_id = 'detector-ycbv_stairs--720976'
        coarse_run_id = 'ycbv_stairs-coarse-4GPU-fixed-434372'
        refiner_run_id = 'ycbv_stairs-refiner-4GPU-4863'

    # camera parametrization
    cam_infos = json.loads((data_path / 'realsense_d435.json').read_text())
    K_ = np.array([[cam_infos['fx'], 0.0, cam_infos['cx']],
            [0.0, cam_infos['fy'], cam_infos['cy']],
            [0.0, 0.0, 1]])
    K = torch.as_tensor(K_)
    K = K.unsqueeze(0)
    K = K.cuda().float()
    TC0 = Transform(np.eye(3), np.zeros(3))
    T0C = TC0.inverse()
    T0C = T0C.toHomogeneousMatrix()
    resolution = (cam_infos['height'], cam_infos['width'])
    camera = dict(T0C=T0C, K=K_, TWC=T0C, resolution=resolution)

    # detector and predictor loading
    detector = load_detector(detector_run_id)
    predictor, mesh_db = load_pose_models(coarse_run_id, refiner_run_id, object_set=object_set)

    previousPose_init = False
    cropping = False
    number_image = len(os.listdir(scene_path))

    # lists for data storage
    frame_id = []
    image_name_list = []
    object_name = []
    pose = []
    bbox = []
    iter_duration = []
    status = []
    detection_score = []

    # plot utils
    video_render = True
    # clean the images folder
    os.system('rm -f /local/users/gsaurel/cosy/cosypose/images/*.png')
    if object_set == 'tless':
        renderer = BulletSceneRenderer('tless.cad', gpu_renderer=True)
    else:
        renderer = BulletSceneRenderer('ycbv', gpu_renderer=True)
    plt = plotter.Plotter()

    # open pickle
    df_init = pd.read_csv(init_path)

    for i in range (1,number_image):
        print('-'*80)
        print(i)
        print('-'*80)
        timer = Timer()
        timer.start()
        
        image_name = f'{str(i).zfill(4)}.png'
        rgb_path = scene_path / image_name
        rgb = np.array(Image.open(rgb_path))

        # init formatting
        id_min = df_init['id'][df_init['id'].sub(i-1).abs().idxmin()]
        df_frame = df_init.loc[df_init['id'] == id_min].reset_index()
        poses_frame = np.array([np.array(list(map(float,pose.split()))) for pose in df_frame['pose'].values])
        poses_frame = [np.reshape(pose, (4,4)) for pose in poses_frame]
        object_coords = [list(map(float,coord.split())) for coord in df_frame['image_coord'].values]
        object_coords = torch.tensor(object_coords).float().cuda()
        poses_init = torch.tensor(poses_frame).float().cuda()
        pose_init = tc.PandasTensorCollection(infos=df_frame['object_id'], poses=poses_init, object_coords=object_coords)

        detections, final_preds, all_preds = inference2(
                detector, predictor, rgb, K,
                poses_init = pose_init,
                n_coarse_iterations=n_coarse_iterations,
                n_refiner_iterations=n_refiner_iterations)
        
        print(detections.infos)
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
            TCO_init = tc.PandasTensorCollection(infos=detections.infos, poses=final_preds.poses)

        #if len(detections.infos) != 1:
        #    previousPose_init = False

        delta_t = timer.stop()
        print(f'Prediction performed in {delta_t} s.')
        
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
        if (i<10000):
            if video_render:
                plotIm = plt.plot_image(rgb)
                figures = sv_viewer.make_singleview_custom_plot(rgb, camera,
                    renderer, final_preds, detections)
                export_png(figures['pred_overlay'], filename=f'images/{image_name}')

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
                "status": status
            }
        )

    df.to_pickle('results.pkl', protocol=2)


    
if __name__ == '__main__':
    main()
