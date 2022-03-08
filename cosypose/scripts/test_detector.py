from cosypose.utils.tqdm import patch_tqdm; patch_tqdm()  # noqa
import torch.multiprocessing
import time
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
from PIL import Image
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
from cosypose.evaluation.runner_utils import format_results, gather_predictions
from cosypose.utils.distributed import get_rank


from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset
from cosypose.datasets.bop import remap_bop_targets
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper
from cosypose.datasets.samplers import ListSampler

def load_detector(run_id):
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    model = Detector(model)
    return model

def main():
    # path initialization
    data_path = LOCAL_DATA_DIR / 'bop_datasets/tless_stairs/train_pbr' 
    scene_path = data_path  / '000001/rgb'
    rgb_path = LOCAL_DATA_DIR  / 'stairs_photo' / 'stairs_far3.png'
    
    detector_run_id = 'detector-ycbv_stairs--834720'

    # detector and predictor loading
    detector = load_detector(detector_run_id)
    
    # load image for render
    rgb = np.array(Image.open(rgb_path))
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    rgb = rgb[..., :3]
    h, w = rgb.shape[:2]
    rgb = torch.as_tensor(rgb)
    rgb = rgb.unsqueeze(0)
    rgb = rgb.cuda().float().permute(0, 3, 1, 2) / 255

    # plotting
    plt = plotter.Plotter()
    
    # detection 
    detections = detector.get_detections(images=rgb,
            one_instance_per_class=False,
            detection_th = 0.9,
            output_masks = None,
            mask_th = 0.5)
    
    # render
    rgb = np.array(Image.open(rgb_path))
    plotIm = plt.plot_image(rgb)
    figures = plt.plot_maskrcnn_bboxes(plotIm, detections)
    export_png(figures, filename='results.png')

    
if __name__ == '__main__':
    main()
