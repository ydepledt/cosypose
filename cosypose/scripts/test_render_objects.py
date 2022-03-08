import numpy as np
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.config import LOCAL_DATA_DIR
from tqdm import tqdm
import torch
from PIL import Image


if __name__ == '__main__':
    # obj_ds_name = 'hb'
    obj_ds_name = 'ycbv_stairs'

    # test bullet scene renderer
    renderer = BulletSceneRenderer(obj_ds_name, gpu_renderer=True)
    TCO = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 0, 1]
    ]).numpy()

    fx, fy = 980, 980
    cx, cy = 378, 48
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,   0,  1]
    ])
    cam = dict(
        resolution=(640, 480),
        K=K,
        TWC=np.eye(4)
    )

    all_images = []
    labels = renderer.urdf_ds.index['label'].tolist()
    label = 'obj_000001'
    obj = dict(
            name=label,
            TWO=TCO)

    # test batch renderer
    #renderer_batch = BulletBatchRenderer(object_set=obj_ds_name, n_workers=0, preload_cache=False)
    #renders_batch = renderer_batch.render(obj_infos=[dict(name=label)], TCO=[TCO], K=[K], resolution=(480,640))
    debug_data = LOCAL_DATA_DIR / 'debug_data' / 'debug_iter=1.pth.tar'
    chck = torch.load(debug_data)
    renders_batch = chck['renders']
    
    im_batch = renders_batch[12]
    im_batch = im_batch.permute(1,2,0)*255
    im_batch = im_batch.cpu().numpy()
    im_batch = im_batch.astype(np.uint8)
    im_batch = Image.fromarray(im_batch)
    im_batch = im_batch.convert('RGB')
    im_batch.save('results_batch.png')



    renders = renderer.render_scene([obj], [cam])[0]['rgb']
    print(renders.shape)
    print(renders)
    im = Image.fromarray(renders)
    im.save('results.png')
        
