#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Automatic hinge detection for articulated objects"""

import numpy as np
from scipy.spatial import cKDTree
import trimesh


class HingeDetector:
    """Detect hinge position and axis from segmented parts"""
    
    def __init__(self, lid_mesh: trimesh.Trimesh, body_mesh: trimesh.Trimesh):
        """
        Initialize detector with two mesh parts
        
        Args:
            lid_mesh: The movable part (lid)
            body_mesh: The fixed part (body)
        """
        self.lid = lid_mesh
        self.body = body_mesh
        
        self.hinge_position = None
        self.hinge_axis = None
    
    def detect(self, threshold=0.01) -> dict:
        """
        Detect hinge position and axis
        
        Args:
            threshold: Distance threshold for contact detection (default: 1cm)
        
        Returns:
            dict with keys: position, axis, contact_vertices_lid, contact_vertices_body
        """
        lid_vertices = self.lid.vertices
        body_vertices = self.body.vertices
        
        # Check if meshes are valid
        if len(lid_vertices) == 0:
            raise ValueError("Lid mesh is empty! Check segmentation results.")
        if len(body_vertices) == 0:
            raise ValueError("Body mesh is empty! Check segmentation results.")
        
        print(f"  Lid vertices: {len(lid_vertices):,}")
        print(f"  Body vertices: {len(body_vertices):,}")
        
        # Build KD-tree for efficient nearest neighbor search
        body_tree = cKDTree(body_vertices)
        
        # Find nearest distances from lid to body
        distances, _ = body_tree.query(lid_vertices)
        min_dist = np.min(distances)
        
        # Define contact threshold
        contact_threshold = min_dist + threshold
        
        # Get contact vertices on lid
        lid_contact_mask = distances < contact_threshold
        lid_contact_vertices = lid_vertices[lid_contact_mask]
        
        # Get contact vertices on body
        lid_tree = cKDTree(lid_vertices)
        distances_body, _ = lid_tree.query(body_vertices)
        body_contact_mask = distances_body < contact_threshold
        body_contact_vertices = body_vertices[body_contact_mask]
        
        if len(lid_contact_vertices) == 0 or len(body_contact_vertices) == 0:
            raise ValueError("No contact region found between parts")
        
        # Compute hinge center as average of contact regions
        lid_contact_center = np.mean(lid_contact_vertices, axis=0)
        body_contact_center = np.mean(body_contact_vertices, axis=0)
        self.hinge_position = (lid_contact_center + body_contact_center) / 2
        
        # Compute hinge axis using PCA on contact vertices
        all_contact = np.vstack([lid_contact_vertices, body_contact_vertices])
        contact_center = np.mean(all_contact, axis=0)
        centered_contact = all_contact - contact_center
        
        # Principal Component Analysis
        cov = np.cov(centered_contact.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Largest eigenvalue corresponds to hinge axis
        max_idx = np.argmax(eigenvalues)
        self.hinge_axis = eigenvectors[:, max_idx].real
        self.hinge_axis = self.hinge_axis / np.linalg.norm(self.hinge_axis)
        
        # If axis is not dominant, use default
        axis_confidence = eigenvalues[max_idx] / np.sum(eigenvalues)
        if axis_confidence < 0.5:
            print(f"Warning: Hinge axis uncertain (confidence={axis_confidence:.2%}), using default X-axis")
            self.hinge_axis = np.array([1, 0, 0])
        
        return {
            'position': self.hinge_position,
            'axis': self.hinge_axis,
            'contact_vertices_lid': lid_contact_vertices,
            'contact_vertices_body': body_contact_vertices,
            'min_distance': min_dist,
            'axis_confidence': axis_confidence
        }
    
    def get_translation_to_origin(self) -> np.ndarray:
        """Get translation vector to move hinge to origin"""
        if self.hinge_position is None:
            raise ValueError("Must call detect() first")
        return -self.hinge_position


