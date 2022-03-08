from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from bokeh.io import export_png
from bokeh.layouts import grid, gridplot
from cosypose.datasets.datasets_cfg import make_scene_dataset
from cosypose.visualization.plotter import Plotter

def main():
    ds_name = 'synthetic.tless_test-10.train'
    scene_ds = make_scene_dataset(ds_name)
    
    plotter = Plotter()
    figures = []
    ids = np.random.randint(len(scene_ds), size=9)
    for idx in ids:
        im, mask, obs = scene_ds[idx]
        im = np.asarray(im)[..., :3]
        f = plotter.plot_image(im)
        export_png(f, filename=f'images/results{idx}.png')

if __name__ == '__main__':
    main()
        
