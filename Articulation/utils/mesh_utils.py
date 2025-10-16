#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mesh loading and manipulation utilities"""

import trimesh
import numpy as np


def load_mesh(path: str) -> trimesh.Trimesh:
    """
    Load a mesh from file, handling both single meshes and scenes
    
    Args:
        path: Path to mesh file (.glb, .obj, .stl, etc.)
    
    Returns:
        trimesh.Trimesh: Loaded and potentially merged mesh
    """
    mesh = trimesh.load(path, force='mesh')
    
    # If it's a Scene with multiple geometries, merge them
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError(f"No valid meshes found in scene: {path}")
    
    return mesh


def save_mesh(mesh: trimesh.Trimesh, path: str):
    """
    Save a mesh to file
    
    Args:
        mesh: Trimesh object to save
        path: Output file path
    """
    mesh.export(path)


def get_mesh_bounds(mesh: trimesh.Trimesh) -> tuple:
    """
    Get bounding box of mesh
    
    Returns:
        tuple: (min_point, max_point)
    """
    return mesh.bounds[0], mesh.bounds[1]


def translate_mesh(mesh: trimesh.Trimesh, translation: np.ndarray) -> trimesh.Trimesh:
    """
    Translate mesh by given vector
    
    Args:
        mesh: Input mesh
        translation: 3D translation vector
    
    Returns:
        Translated mesh
    """
    mesh_copy = mesh.copy()
    mesh_copy.apply_translation(translation)
    return mesh_copy


def compute_mesh_center(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Compute center of mesh bounding box
    
    Returns:
        np.ndarray: Center point (3,)
    """
    bounds = mesh.bounds
    return (bounds[0] + bounds[1]) / 2


