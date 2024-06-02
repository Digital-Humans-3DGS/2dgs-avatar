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
import math
import sys
sys.path.append('/home/u200110727/dh/diff-gaussian-rasterization')

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def render(data,
           iteration,
           scene,
           pipe,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           compute_loss=True,
           return_opacity=False, ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    pc, loss_reg, colors_precomp = scene.convert_gaussians(data, iteration, compute_loss)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    #######################
    # print(data.world_view_transform.shape)
    # print(data.world_view_transform)

    # Set up rasterization configuration
    tanfovx = math.tan(data.FoVx * 0.5)
    tanfovy = math.tan(data.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data.image_height),
        image_width=int(data.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier, 
        viewmatrix=data.world_view_transform, 
        projmatrix=data.full_proj_transform, 
        sh_degree=pc.active_sh_degree,
        campos=data.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # depth
    # Ensure pc.get_xyz and data.camera_center are tensors
    points = pc.get_xyz

    # Compute the Euclidean distance between xyz and camera_center
    # distance = torch.norm(xyz - camera_center, dim=1).unsqueeze(1)  # Shape: [17802]

    world_view_transform = data.world_view_transform  # Shape: [4, 4]

    
    # 添加一列全为1的值，扩展points为齐次坐标
    ones = torch.ones(points.shape[0], 1).cuda()  # Shape: [N, 1]
    homogeneous_points = torch.cat([points, ones], dim=1)  # Shape: [N, 4]

    # transformed_points = torch.matmul(homogeneous_points, world_view_transform.transpose(0, 1))  # Shape: [N, 4]
    # transformed_points = torch.matmul(world_view_transform.T, homogeneous_points.T).T  # Shape: [N, 4]
    transformed_points = torch.matmul(homogeneous_points,world_view_transform)  # Shape: [N, 4]
    # print(transformed_points[:, 2].min())

    # sys.exit()

    z_values = transformed_points[:, 2]  # Shape: [N]

    # print(world_view_transform)
    # print(points[0])

    # print(z_values[0])

    # sys.exit()
    # assert z_values > 0

    

    # 将 points 转换为齐次坐标
    # ones = torch.ones(points.shape[0], 1, device=points.device)
    # points_homogeneous = torch.cat([points, ones], dim=1)  # 形状为 (N, 4)

    # # 应用视图矩阵进行变换
    # points_transformed_homogeneous = torch.matmul(world_view_transform, points_homogeneous.T).T

    # # 将结果转换为三维坐标，去除齐次坐标的最后一个分量
    # points_transformed = points_transformed_homogeneous[:, :3] / points_transformed_homogeneous[:, 3].unsqueeze(1)

    # z_values = points_transformed[..., 2:].expand(-1, 3)

    # print(z_values)

    # Normalize the distance tensor along dimension 0
    # normalized_distance = distance.unsqueeze(1)
    # normalized_distance = torch.nn.functional.normalize(distance.unsqueeze(1), dim=0)
    # min_distance = distance.min()
    # max_distance = distance.max()
    # normalized_distance = ((distance - min_distance) / (max_distance - min_distance)).unsqueeze(1) 

    # print(normalized_distance.shape)

    # Expand the dimensions to match the required shape
    expanded_distance = z_values.unsqueeze(1).expand(-1, 3)
    # expanded_distance = z_values


    # Convert to depth color by scaling with 255
    depth_color = expanded_distance 

    depth_image, _, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = depth_color,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # print("WARNING: depth min max: ", depth_image.min(), depth_image.max())

    opacity_image = None
    if return_opacity:
        opacity_image, _, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=torch.ones(opacity.shape[0], 3, device=opacity.device),
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        opacity_image = opacity_image[:1]


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"deformed_gaussian": pc,
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "loss_reg": loss_reg,
            "opacity_render": opacity_image,
            "depth": depth_image,
            }
