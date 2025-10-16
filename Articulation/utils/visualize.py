#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize segmentation and URDF results"""

import sys
import os
import json
import numpy as np
import trimesh
import open3d as o3d
from pathlib import Path


def visualize_urdf_output(urdf_dir: str):
    """
    Visualize URDF output directory
    
    Shows:
    - Aligned parts (lid and body)
    - Hinge position at origin
    - Hinge axis
    """
    
    print(f"Visualizing URDF output: {urdf_dir}")
    
    # Load metadata
    metadata_path = os.path.join(urdf_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.json not found in {urdf_dir}")
        return 1
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load meshes
    lid_path = os.path.join(urdf_dir, "lid_centered.glb")
    body_path = os.path.join(urdf_dir, "body_centered.glb")
    
    if not os.path.exists(lid_path) or not os.path.exists(body_path):
        print("Error: Mesh files not found")
        return 1
    
    lid = trimesh.load(lid_path, force='mesh')
    body = trimesh.load(body_path, force='mesh')
    
    if isinstance(lid, trimesh.Scene):
        lid = trimesh.util.concatenate([g for g in lid.geometry.values()])
    if isinstance(body, trimesh.Scene):
        body = trimesh.util.concatenate([g for g in body.geometry.values()])
    
    # Color meshes
    lid.visual.vertex_colors = [100, 200, 100, 255]  # Green
    body.visual.vertex_colors = [200, 150, 100, 255]  # Brown
    
    # Create coordinate axes at origin
    axis_length = 0.3
    
    # X axis (red)
    x_axis = trimesh.creation.cylinder(
        radius=0.01,
        segment=[[0, 0, 0], [axis_length, 0, 0]]
    )
    x_axis.visual.vertex_colors = [255, 0, 0, 255]
    
    # Y axis (green)
    y_axis = trimesh.creation.cylinder(
        radius=0.01,
        segment=[[0, 0, 0], [0, axis_length, 0]]
    )
    y_axis.visual.vertex_colors = [0, 255, 0, 255]
    
    # Z axis (blue)
    z_axis = trimesh.creation.cylinder(
        radius=0.01,
        segment=[[0, 0, 0], [0, 0, axis_length]]
    )
    z_axis.visual.vertex_colors = [0, 0, 255, 255]
    
    # Hinge axis (yellow, thicker)
    hinge_axis = np.array(metadata['hinge']['axis'])
    hinge_vis = trimesh.creation.cylinder(
        radius=0.02,
        segment=[-hinge_axis * 0.5, hinge_axis * 0.5]
    )
    hinge_vis.visual.vertex_colors = [255, 255, 0, 255]
    
    # Origin sphere (white)
    origin = trimesh.creation.icosphere(radius=0.03)
    origin.visual.vertex_colors = [255, 255, 255, 255]
    
    # Create scene
    scene = trimesh.Scene([lid, body, x_axis, y_axis, z_axis, hinge_vis, origin])
    
    print("\n" + "="*70)
    print("Visualization Guide:")
    print("="*70)
    print("  • Green mesh = Lid (movable part)")
    print("  • Brown mesh = Body (fixed part)")
    print("  • Red axis = X")
    print("  • Green axis = Y")
    print("  • Blue axis = Z")
    print("  • Yellow thick axis = Hinge axis")
    print("  • White sphere = Origin (hinge position)")
    print("="*70)
    print(f"\nHinge Information:")
    print(f"  Original position: {metadata['hinge']['original_position']}")
    print(f"  Axis: {metadata['hinge']['axis']}")
    print(f"  Axis confidence: {metadata['hinge']['axis_confidence']:.2%}")
    print(f"\nJoint Limits:")
    print(f"  {metadata['joint_limits']['lower']:.2f} - {metadata['joint_limits']['upper']:.2f} rad")
    print(f"  ({metadata['joint_limits']['lower_deg']:.1f}° - {metadata['joint_limits']['upper_deg']:.1f}°)")
    print("="*70 + "\n")
    
    # Show scene
    scene.show()
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m utils.visualize <urdf_output_directory>")
        sys.exit(1)
    
    sys.exit(visualize_urdf_output(sys.argv[1]))


