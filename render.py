#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm, trange
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import fix_random
from scene import GaussianModel

from scene.cameras import Camera


from utils.general_utils import Evaluator, PSEvaluator

import hydra
from omegaconf import OmegaConf
import wandb
# new
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
import open3d as o3d

DEBUG=True

def predict(config):
    with torch.set_grad_enabled(False):
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        times = []
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            iter_start.record()

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
                                compute_loss=False, return_opacity=False)
            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),]
            wandb.log({'test_images': wandb_img})

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))

            # evaluate
            times.append(elapsed)

        _time = np.mean(times[1:])
        wandb.log({'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 time=_time)



def test(config):
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        evaluator = PSEvaluator() if config.dataset.name == 'people_snapshot' else Evaluator()

        psnrs = []
        ssims = []
        lpipss = []
        times = []
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            iter_start.record()

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
                                compute_loss=False, return_opacity=False)

            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            gt = view.original_image[:3, :, :]

            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),
                        wandb.Image(gt[None], caption='gt_{}'.format(view.image_name))]

            wandb.log({'test_images': wandb_img})

            # print("export mesh ...")
            gaussExtractor = GaussianExtractor(scene, render, config.opt.iterations, config.pipeline, background=background)    
            os.makedirs(render_path, exist_ok=True)
            # set the active_sh to 0 to export only diffuse texture
            gaussExtractor.gaussians.active_sh_degree = 0

            cameras = view.all_cameras

            views = []
            for k, camera in cameras.items():
                K = np.array(camera['K'], dtype=np.float32).copy()
                dist = np.array(camera['D'], dtype=np.float32).ravel()
                R = np.array(camera['R'], np.float32)
                T = np.array(camera['T'], np.float32)

                H, W = 1024, 1024

                M = np.eye(3)
                M[0, 2] = (K[0, 2] - W / 2) / K[0, 0]
                M[1, 2] = (K[1, 2] - H / 2) / K[1, 1]
                K[0, 2] = W / 2
                K[1, 2] = H / 2
                R = M @ R
                T = M @ T

                R = np.transpose(R)
                T = T[:, 0]

                K[0, :] *= config.dataset.w / W
                K[1, :] *= config.dataset.h / H

                view_clone = Camera(
                    frame_id=view.frame_id,
                    cam_id=view.cam_id,
                    K=K, R=R, T=T,
                    FoVx=view.FoVx,
                    FoVy=view.FoVy,
                    image=view.image,
                    mask=view.mask,
                    gt_alpha_mask=None,
                    image_name=f"c{view.cam_id}_f{view.frame_id if view.frame_id >= 0 else -view.frame_id - 1:06d}",
                    data_device=config.dataset.data_device,
                    # human params
                    rots=view.rots,
                    Jtrs=view.Jtrs,
                    bone_transforms=view.bone_transforms,
                    all_cameras=None,
                )
                views.append(view_clone)

            # reconstruct the mesh
            gaussExtractor.reconstruction(views)


            # extract the mesh and save
            name = f'fuse{idx}.ply'
            mesh_res = 1024
            depth_trunc = 10
            num_cluster = 1000
            voxel_size = 0.004
            sdf_trunc = 0.02
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
            o3d.io.write_triangle_mesh(os.path.join(render_path, name), mesh)
            # print("mesh saved at {}".format(os.path.join(render_path, name)))
            # post-process the mesh and save, saving the largest N clusters
            # mesh_post = post_process_mesh(mesh, cluster_to_keep=num_cluster)
            # o3d.io.write_triangle_mesh(os.path.join(render_path, name.replace('.ply', '_post.ply')), mesh_post)
            # print("mesh post processed saved at {}".format(os.path.join(render_path, name.replace('.ply', '_post.ply'))))
        
            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))

            # evaluate
            if config.evaluate:
                metrics = evaluator(rendering, gt)
                psnrs.append(metrics['psnr'])
                ssims.append(metrics['ssim'])
                lpipss.append(metrics['lpips'])
            else:
                psnrs.append(torch.tensor([0.], device='cuda'))
                ssims.append(torch.tensor([0.], device='cuda'))
                lpipss.append(torch.tensor([0.], device='cuda'))
            times.append(elapsed)

        _psnr = torch.mean(torch.stack(psnrs))
        _ssim = torch.mean(torch.stack(ssims))
        _lpips = torch.mean(torch.stack(lpipss))
        _time = np.mean(times[1:])
        
        
        wandb.log({'metrics/psnr': _psnr,
                'metrics/ssim': _ssim,
                'metrics/lpips': _lpips,
                'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                psnr=_psnr.cpu().numpy(),
                ssim=_ssim.cpu().numpy(),
                lpips=_lpips.cpu().numpy(),
                time=_time)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)

    # set wandb logger
    if config.mode == 'test':
        config.suffix = config.mode + '-' + config.dataset.test_mode
    elif config.mode == 'predict':
        predict_seq = config.dataset.predict_seq
        if config.dataset.name == 'zjumocap':
            predict_dict = {
                0: 'dance0',
                1: 'dance1',
                2: 'flipping',
                3: 'canonical'
            }
        else:
            predict_dict = {
                0: 'rotation',
                1: 'dance2',
            }
        predict_mode = predict_dict[predict_seq]
        config.suffix = config.mode + '-' + predict_mode
    else:
        raise ValueError
    if config.dataset.freeview:
        config.suffix = config.suffix + '-freeview'
    wandb_name = config.name + '-' + config.suffix
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='gaussian-splatting-avatar-test',
        entity='iron-lu',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    fix_random(config.seed)

    if config.mode == 'test':
        test(config)
    elif config.mode == 'predict':
        predict(config)
    else:
        raise ValueError

if __name__ == "__main__":
    main()