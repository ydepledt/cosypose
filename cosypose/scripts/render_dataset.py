
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.datasets.datasets_cfg import make_scene_dataset
from bokeh.io import export_png
from bokeh.plotting import gridplot
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from cosypose.lib3d import Transform
from cosypose.visualization.plotter import Plotter
import os

def get_random_image(scene_ds, idx=None):
    renderer = BulletSceneRenderer(urdf_ds_name)
    if idx is None:
        idx = np.random.randint(len(scene_ds))
    print("idx:", idx)
    ds_rgb, mask, state = scene_ds[idx]
                                    
    objects = state['objects']
    camera = state['camera']
    cameras = [camera]
    image = renderer.render_scene(objects, cameras)[0]['rgb']
    renderer.disconnect()
    return ds_rgb, image

if __name__ == '__main__':
    ds_name, urdf_ds_name = 'ycbv_stairs.train.pbr', 'ycbv_stairs'
    scene_ds = make_scene_dataset(ds_name)
    ds_rgb, render_rgb = get_random_image(scene_ds)
    plotter = Plotter()
    fig_ds_rgb = plotter.plot_image(ds_rgb)
    fig_render_rgb = plotter.plot_image(render_rgb)
    fig_overlay = plotter.plot_overlay(ds_rgb, render_rgb)
    export_png(fig_overlay, filename = 'results.png')
