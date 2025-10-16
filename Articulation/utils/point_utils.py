#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Point cloud projection and manipulation utilities"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_min
import cv2


def project_pcd(pnt_w, K, c2w):
    """
    Project 3D points to 2D image coordinates
    
    Args:
        pnt_w: World coordinates of points (N, 3)
        K: Camera intrinsic matrix (3, 3)
        c2w: Camera-to-world transform (4, 4)
    
    Returns:
        tuple: (uv_cam, pnt_cam, depth)
    """
    cam_pos = c2w[:3, 3]
    pnt_cam = ((pnt_w - cam_pos)[..., None, :] * c2w[:3, :3].transpose()).sum(-1)
    uv_cam = (pnt_cam / pnt_cam[..., 2:]) @ K.T
    return uv_cam, pnt_cam, pnt_cam[..., 2:]


def unproject_pcd(pnt_cam, c2w):
    """
    Unproject camera coordinates back to world coordinates
    
    Args:
        pnt_cam: Camera coordinates (N, 3)
        c2w: Camera-to-world transform (4, 4)
    
    Returns:
        World coordinates (N, 3)
    """
    pnt_w = c2w[:3, :3] @ pnt_cam.T + c2w[:3, 3, np.newaxis]
    return pnt_w.transpose()


def get_depth_map(uv, depth, h, w, bg_depth=1e10, scale=2):
    """
    Create depth map from point cloud projections
    
    Args:
        uv: Image coordinates (N, 2)
        depth: Depth values (N,)
        h, w: Image dimensions
        bg_depth: Background depth value
        scale: Downscaling factor for efficiency
    
    Returns:
        tuple: (depth_map, index)
    """
    _h, _w = int(h / scale), int(w / scale)
    uv = (uv / scale).round().astype(np.int32)[..., :2]
    uv = uv.clip(0, np.array([_w, _h]) - 1)
    
    depth_ten = torch.from_numpy(depth).float().reshape(-1)
    uv_int_flat_ten = torch.from_numpy(uv[:, 0] * _h + uv[:, 1]).long()
    depth_map_ten = torch.ones((_w, _h, 1), dtype=torch.float32).reshape(-1) * bg_depth
    
    # Scatter min to handle overlapping points
    _, index = scatter_min(depth_ten, uv_int_flat_ten, dim=0, out=depth_map_ten)
    depth_map = depth_map_ten.reshape(_w, _h).transpose(0, 1).numpy()
    
    # Upscale back to original resolution
    depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth_map, index


def mask_pcd_2d(uv, mask, thresh=0.5, depth=None, pnt_depth=None, depth_thresh=0.1):
    """
    Apply 2D mask to point cloud
    
    Args:
        uv: Image coordinates (N, 2)
        mask: 2D binary mask (H, W)
        thresh: Threshold for mask sampling
        depth: Optional depth map for depth filtering
        pnt_depth: Optional point depths
        depth_thresh: Depth difference threshold
    
    Returns:
        Binary mask for points (N,)
    """
    h, w = mask.shape
    h_w_half = np.array([w, h]) / 2
    uv_rescale = (uv[:, :2] - h_w_half) / h_w_half
    uv_rescale_ten = torch.from_numpy(uv_rescale).float()[None, None, ...]
    mask_ten = torch.from_numpy(mask).float()[None, None, ...]
    
    sample_mask = F.grid_sample(
        mask_ten, uv_rescale_ten,
        padding_mode="border", align_corners=True
    ).reshape(-1).numpy()
    
    sample_mask = sample_mask > thresh
    
    if depth is not None and pnt_depth is not None:
        sample_depth = F.grid_sample(
            torch.from_numpy(depth).float()[None, None, ...], uv_rescale_ten,
            padding_mode="border", align_corners=True
        ).reshape(-1).numpy()
        sample_mask = sample_mask & (np.abs(sample_depth - pnt_depth[..., 0]) < depth_thresh)
    
    return sample_mask


