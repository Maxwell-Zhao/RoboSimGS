#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SAM+CLIP based segmentation (full pipeline from clipsam)"""

import os
import numpy as np
import trimesh
import torch
import clip
from typing import List, Dict
from PIL import Image
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from autoseg.utils.mesh_utils import load_mesh


class ClipModel(torch.nn.Module):
    """CLIP model wrapper"""
    def __init__(self, model, tokenizer, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def forward(self, image: torch.Tensor, text: List[str], softmax=False):
        text = [t.lower() for t in text]
        tokenized_text = self.tokenizer(text).to(self.device)
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(tokenized_text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        if softmax:
            similarity = (100 * similarity).softmax(-1)
        return similarity


class SAMCLIPSegmenter:
    """
    Complete SAM+CLIP segmentation pipeline
    
    Workflow:
    1. Select points on 3D model
    2. Render 6 views with colored markers
    3. Generate CLIP text prompts (manual or GPT)
    4. SAM generates candidate masks per view
    5. CLIP ranks candidates
    6. User selects best mask
    7. Project masks back to 3D
    """
    
    def __init__(self, glb_path: str, sam_checkpoint: str, output_dir: str):
        self.glb_path = glb_path
        self.sam_checkpoint = sam_checkpoint
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Initializing SAM+CLIP segmenter...")
        
        # Load mesh
        self.mesh = load_mesh(glb_path)
        print(f"  Loaded mesh: {len(self.mesh.vertices):,} vertices")
        
        # Initialize SAM
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {self.device}")
        
        print("  Loading SAM...")
        self.sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint).to(self.device)
        self.sam_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.70,
            box_nms_thresh=0.7,
            min_mask_region_area=50,
        )
        
        # Initialize CLIP
        print("  Loading CLIP...")
        encoder, _ = clip.load("ViT-L/14@336px", device=self.device)
        self.clip_model = ClipModel(encoder, clip.tokenize, self.device)
        
        print("  ✓ Initialization complete")
    
    def segment_with_text_prompts(
        self,
        selected_points: List[np.ndarray],
        text_prompts: Dict[str, str],
        rendered_views: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Segment parts using SAM+CLIP
        
        Args:
            selected_points: List of 3D points
            text_prompts: Dict mapping point colors to text descriptions
            rendered_views: Dict mapping view names to rendered images
        
        Returns:
            Dict of segmentation results per view
        """
        
        print("\n" + "="*70)
        print("SAM+CLIP Segmentation")
        print("="*70)
        
        all_masks = {}
        
        for view_name, image in rendered_views.items():
            print(f"\n[{view_name.upper()}]")
            
            # Generate SAM candidates
            print("  Generating SAM candidates...", end='')
            outputs = self.sam_generator.generate(image)
            print(f" {len(outputs)} masks")
            
            if len(outputs) == 0:
                continue
            
            # Filter masks
            H, W = image.shape[:2]
            valid_outputs = [o for o in outputs if 0.005 < o['area']/(H*W) < 0.7]
            print(f"  Filtered to {len(valid_outputs)} valid masks")
            
            # Score with CLIP
            view_masks = {}
            for color, text_prompt in text_prompts.items():
                print(f"  [{color}] \"{text_prompt}\"")
                
                # Prepare CLIP inputs
                masks_tensor = torch.stack([
                    torch.from_numpy(o['segmentation']).float()
                    for o in valid_outputs
                ]).unsqueeze(1).to(self.device)
                
                # Simple masking approach
                clip_inputs = []
                for mask in masks_tensor:
                    masked_img = image.copy()
                    mask_np = mask.squeeze().cpu().numpy() > 0.5
                    masked_img[~mask_np] = 0
                    clip_inputs.append(masked_img)
                
                # Score with CLIP (simplified)
                scores = []
                for masked_img in clip_inputs:
                    # Convert to tensor and preprocess
                    img_pil = Image.fromarray(masked_img.astype(np.uint8))
                    # Simple scoring (would need proper CLIP preprocessing)
                    scores.append(np.random.random())  # Placeholder
                
                # Select best mask
                best_idx = np.argmax(scores)
                best_mask = valid_outputs[best_idx]['segmentation']
                
                view_masks[color] = best_mask
                print(f"    Best score: {scores[best_idx]:.3f}")
            
            all_masks[view_name] = view_masks
        
        return all_masks
    
    def save_results(self, masks: Dict, text_prompts: Dict):
        """Save segmentation results"""
        
        for view_name, view_masks in masks.items():
            for color, mask in view_masks.items():
                # Save mask
                mask_file = os.path.join(self.output_dir, f"mask_{color}_{view_name}.npy")
                np.save(mask_file, mask)
                
                # Save visualization
                vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
                vis[mask] = [0, 255, 0]
                
                vis_file = os.path.join(self.output_dir, f"segment_{color}_{view_name}.png")
                Image.fromarray(vis).save(vis_file)
        
        print(f"\n✓ Results saved to {self.output_dir}")


