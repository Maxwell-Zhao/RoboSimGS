#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration management for the pipeline"""

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """Pipeline configuration"""
    
    input_file: str
    output_dir: str = None
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    robot_name: str = "articulated_object"
    verbose: bool = False
    
    def __post_init__(self):
        # Set default output directory
        if self.output_dir is None:
            input_path = Path(self.input_file)
            self.output_dir = str(input_path.parent / f"{input_path.stem}_output")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subdirectories
        self.segmentation_dir = os.path.join(self.output_dir, "segmentation")
        self.parts_dir = os.path.join(self.output_dir, "parts")
        self.urdf_dir = os.path.join(self.output_dir, "urdf")
        
        os.makedirs(self.segmentation_dir, exist_ok=True)
        os.makedirs(self.parts_dir, exist_ok=True)
        os.makedirs(self.urdf_dir, exist_ok=True)
    
    @property
    def sam_checkpoint_path(self):
        """Get full path to SAM checkpoint"""
        if os.path.isabs(self.sam_checkpoint):
            return self.sam_checkpoint
        
        # Check in current directory
        if os.path.exists(self.sam_checkpoint):
            return self.sam_checkpoint
        
        # Check in project root
        project_root = Path(__file__).parent.parent
        checkpoint_path = project_root / self.sam_checkpoint
        if checkpoint_path.exists():
            return str(checkpoint_path)
        
        raise FileNotFoundError(f"SAM checkpoint not found: {self.sam_checkpoint}")


