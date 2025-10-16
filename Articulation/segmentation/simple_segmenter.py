#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified SAM-based segmentation without CLIP
Direct mask selection and mesh splitting
"""

import os
import sys
import numpy as np
import trimesh
import open3d as o3d
import torch
from typing import List, Dict
import json
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import base64
import io

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from autoseg.utils.mesh_utils import load_mesh, save_mesh

# Colors for visualization
COLORS = [
    [1.0, 0.0, 0.0],  # Red - Lid
    [0.0, 1.0, 0.0],  # Green - Body
]

COLOR_NAMES = ["RED", "GREEN"]

try:
    import openai
    HAS_OPENAI = True
except:
    HAS_OPENAI = False


class SimpleSegmenter:
    """
    Simplified segmentation pipeline:
    1. Select points to mark lid and body
    2. Render top/bottom views with/without markers
    3. SAM generates all candidate masks (on clean images)
    4. User/GPT selects which mask is lid and which is body
    5. Split mesh directly based on 2D masks
    """
    
    def __init__(self, glb_path: str, sam_checkpoint: str, sam_model_type: str = 'vit_h', output_dir: str = None):
        print("\n" + "="*70)
        print("Simplified SAM Segmentation Pipeline")
        print("="*70)
        
        self.glb_path = glb_path
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Load mesh
        print("\nLoading 3D model...")
        self.mesh = load_mesh(glb_path)
        print(f"  Vertices: {len(self.mesh.vertices):,}")
        print(f"  Faces: {len(self.mesh.faces):,}")
        
        # Convert to Open3D for visualization
        self.o3d_mesh = o3d.geometry.TriangleMesh()
        self.o3d_mesh.vertices = o3d.utility.Vector3dVector(self.mesh.vertices)
        self.o3d_mesh.triangles = o3d.utility.Vector3iVector(self.mesh.faces)
        self.o3d_mesh.compute_vertex_normals()
        
        # Mesh properties
        bounds = self.mesh.bounds
        self.center = (bounds[0] + bounds[1]) / 2
        self.size = np.linalg.norm(bounds[1] - bounds[0])
        
        self.selected_points = []
        self.selected_indices = []
        
        # Initialize SAM
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {self.device}")
        
        print(f"  Loading SAM ({sam_model_type})...")
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(self.device)
        self.sam_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.75,
            min_mask_region_area=100,
        )
        
        print("  ✓ Initialization complete\n")
    
    def select_points(self, max_points: int = 2):
        """Select points to mark different parts (typically 2 points: lid and body)"""
        print("\n" + "="*70)
        print("Point Selection")
        print("="*70)
        print("Instructions:")
        print("  • Select points to mark different parts")
        print("  • Typically 2 points: 1 on LID, 1 on BODY")
        print("  • [Shift + Left Click] to select")
        print("  • [Q] or close window to finish")
        print("="*70 + "\n")
        
        # Create point cloud for selection
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.mesh.vertices)
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
        
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(
            window_name="Select Points - [Shift + Click]",
            width=1280, height=720
        )
        vis.add_geometry(pcd)
        
        opt = vis.get_render_option()
        opt.point_size = 10.0
        opt.background_color = np.array([0.1, 0.1, 0.1])
        
        vis.run()
        picked_indices = vis.get_picked_points()
        vis.destroy_window()
        
        if len(picked_indices) < 2:
            raise ValueError("Need at least 2 points (1 for lid, 1 for body)")
        
        if len(picked_indices) > max_points:
            print(f"Selected {len(picked_indices)} points, using first {max_points}")
            picked_indices = picked_indices[:max_points]
        
        # Map to mesh vertices
        pcd_points = np.asarray(pcd.points)
        mesh_vertices = self.mesh.vertices
        
        for idx in picked_indices:
            point = pcd_points[idx]
            distances = np.linalg.norm(mesh_vertices - point, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_point = mesh_vertices[nearest_idx]
            
            self.selected_points.append(nearest_point)
            self.selected_indices.append(int(nearest_idx))
        
        print(f"\n✓ Selected {len(self.selected_points)} points")
        for i, point in enumerate(self.selected_points):
            color = COLOR_NAMES[i]
            print(f"  {i+1}. [{color}] Position: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
        
        return True
    
    def input_part_names(self):
        """Let user input names for each point/part (manual or GPT)"""
        print("\n" + "="*70)
        print("Name Each Part")
        print("="*70)
        print("\nChoose naming method:")
        print("  1. Manual input (you name each part)")
        print("  2. GPT automatic naming (AI names parts)")
        
        choice = input("\nSelect (1/2, default=1): ").strip()
        
        if choice == '2':
            print("\nUsing GPT for automatic naming...")
            # Need to render first for GPT to see the object
            print("  Note: Will render views first for GPT to analyze")
            
            # Quick render for naming (will render again later with better quality)
            temp_views = self._quick_render_for_gpt()
            part_names = self._gpt_generate_part_names(temp_views)
            
            if not part_names or len(part_names) != len(self.selected_points):
                print("  ✗ GPT naming failed, switching to manual input")
                part_names = self._manual_input_part_names()
        else:
            part_names = self._manual_input_part_names()
        
        self.part_names = part_names
        
        print(f"\n{'='*70}")
        print("Part names assigned:")
        for i, (color, name) in enumerate(zip(COLOR_NAMES, self.part_names)):
            print(f"  • {color}: {name}")
        print(f"{'='*70}")
        
        return self.part_names
    
    def _manual_input_part_names(self) -> List[str]:
        """Manual input of part names"""
        print("\n" + "="*70)
        print("Manual Part Naming")
        print("="*70)
        print("\nGive each point a name (e.g., 'lid', 'body', 'handle', etc.)")
        print("="*70)
        
        part_names = []
        
        for i, point in enumerate(self.selected_points):
            color = COLOR_NAMES[i]
            
            print(f"\n[{color}] Point #{i+1}:")
            print(f"  Position: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
            print(f"  Examples: lid, body, cover, base, handle, frame")
            
            while True:
                name = input(f"  Enter part name: ").strip().lower()
                if name:
                    part_names.append(name)
                    print(f"  ✓ {color} = {name}")
                    break
                else:
                    print("  ✗ Cannot be empty")
        
        return part_names
    
    def _quick_render_for_gpt(self, resolution: int = 400):
        """Quick render of 6 marked views for GPT to analyze"""
        print("  Rendering 6 views for GPT naming...")
        
        distance = self.size * 2.0
        view_configs = {
            'front': np.array([0, 0, distance]),
            'back': np.array([0, 0, -distance]),
            'left': np.array([-distance, 0, 0]),
            'right': np.array([distance, 0, 0]),
            'top': np.array([0, distance, 0]),
            'bottom': np.array([0, -distance, 0]),
        }
        
        rendered = {}
        marker_size = 0.03
        
        for view_name, camera_offset in view_configs.items():
            try:
                camera_pos = self.center + camera_offset
                look_at = self.center
                forward = (look_at - camera_pos) / np.linalg.norm(look_at - camera_pos)
                
                if view_name == 'top':
                    up = np.array([0, 0, -1])
                else:
                    up = np.array([0, 1, 0])
                
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward) / np.linalg.norm(np.cross(right, forward))
                
                camera_transform = np.eye(4)
                camera_transform[:3, 0] = right
                camera_transform[:3, 1] = up
                camera_transform[:3, 2] = -forward
                camera_transform[:3, 3] = camera_pos
                
                # Render with markers
                scene = trimesh.Scene([self.mesh])
                
                for i, point in enumerate(self.selected_points):
                    sphere = trimesh.creation.icosphere(subdivisions=3, radius=self.size * marker_size)
                    sphere.apply_translation(point)
                    color_rgb = np.array(COLORS[i]) * 255
                    color_rgba = np.array([color_rgb[0], color_rgb[1], color_rgb[2], 255], dtype=np.uint8)
                    sphere.visual.face_colors = np.tile(color_rgba, (len(sphere.faces), 1))
                    scene.add_geometry(sphere)
                
                scene.camera_transform = camera_transform
                scene.camera.fov = [50, 50]
                
                png = scene.save_image(resolution=[resolution, resolution], visible=True)
                image = np.array(Image.open(trimesh.util.wrap_as_stream(png)))[:, :, :3]
                
                rendered[view_name] = image
                
            except Exception as e:
                print(f"    ✗ {view_name} failed: {e}")
        
        return rendered
    
    def _gpt_generate_part_names(self, marked_views: Dict[str, np.ndarray]) -> List[str]:
        """Use GPT to automatically name parts based on marked views"""
        
        if not HAS_OPENAI:
            print("  ✗ OpenAI not available")
            return []
        
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("  ✗ OPENAI_API_KEY not set")
            return []
        
        openai.api_key = api_key
        
        try:
            # Encode images
            print("  Encoding images...")
            image_contents = []
            for view_name, image in marked_views.items():
                img_pil = Image.fromarray(image)
                buffer = io.BytesIO()
                img_pil.save(buffer, format='PNG')
                buffer.seek(0)
                b64_str = base64.b64encode(buffer.read()).decode()
                
                image_contents.append({
                    "type": "text",
                    "text": f"\n--- {view_name.upper()} VIEW ---"
                })
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_str}", "detail": "high"}
                })
            
            # Build prompt
            color_info = "\n".join([
                f"  - {COLOR_NAMES[i]} marker: Point #{i+1} at {self.selected_points[i]}"
                for i in range(len(self.selected_points))
            ])
            
            prompt = f"""You are analyzing a 3D articulated object with colored markers on different parts.

The object is shown from 6 different viewpoints (FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM).

{len(self.selected_points)} colored markers have been placed:
{color_info}

Your task: 
1. Analyze all 6 views to understand the object structure
2. Identify what part each colored marker is on
3. Give each marker a concise name (1-2 words)

Examples:
- "lid" (for top cover/movable part)
- "body" (for main container/fixed part)  
- "handle" (for grip)
- "base" (for bottom)
- "cover" (for protective layer)

Return ONLY a JSON object mapping colors to part names:
{{
  "RED": "part name",
  "GREEN": "part name"
}}

Use simple, common English words. Return ONLY the JSON, no additional text.
"""

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *image_contents
                ]
            }]
            
            print("  Querying GPT...")
            response = openai.ChatCompletion.create(
                model=os.environ.get('OPENAI_GPT_MODEL', 'gpt-4-turbo'),
                messages=messages,
                max_tokens=200,
                temperature=0.3
            )
            
            gpt_output = response['choices'][0]['message']['content']
            
            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', gpt_output, re.DOTALL)
            if json_match:
                name_map = json.loads(json_match.group())
                
                # Extract names in order
                part_names = []
                for i in range(len(self.selected_points)):
                    color = COLOR_NAMES[i]
                    name = name_map.get(color, f"part{i+1}")
                    part_names.append(name.lower())
                
                print(f"\n  ✓ GPT generated names:")
                for color, name in zip(COLOR_NAMES[:len(part_names)], part_names):
                    print(f"    • {color}: {name}")
                
                return part_names
            else:
                print("  ✗ Failed to parse GPT response")
                return []
                
        except Exception as e:
            print(f"  ✗ GPT call failed: {e}")
            return []
    
    def visualize_selected_points(self):
        """Show selected points with colored spheres"""
        print("\nShowing selected points...")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Confirm Selected Points", width=1024, height=768)
        vis.add_geometry(self.o3d_mesh)
        
        for i, point in enumerate(self.selected_points):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.size * 0.02)
            sphere.translate(point)
            sphere.paint_uniform_color(COLORS[i])
            vis.add_geometry(sphere)
        
        vis.run()
        vis.destroy_window()
    
    def render_views(self, resolution: int = 800, marker_size: float = 0.03):
        """Render oblique top and bottom views"""
        print(f"\nRendering oblique top and bottom views...")
        print(f"  Resolution: {resolution}x{resolution}")
        
        # Oblique angles for better depth perception
        distance = self.size * 2.0
        h = distance  # Vertical distance  
        d = distance * 0.7  # Front/back offset (creates 45-degree angle: atan(0.7/1) ≈ 35°)
        
        view_configs = {
            'top': np.array([0, -h, d]),     # From below, looking up-forward (bottom-front view)
            'bottom': np.array([0, h, d]),   # From above, looking down-forward (top-front view)
        }
        
        rendered_images = {}
        rendered_images_marked = {}
        camera_params = {}
        
        # Camera intrinsics
        fov = 50
        focal_length = (resolution / 2) / np.tan(np.radians(fov / 2))
        intrinsic = np.array([
            [focal_length, 0, resolution / 2],
            [0, focal_length, resolution / 2],
            [0, 0, 1]
        ])
        
        for view_name, camera_offset in view_configs.items():
            try:
                camera_pos = self.center + camera_offset
                look_at = self.center
                forward = (look_at - camera_pos) / np.linalg.norm(look_at - camera_pos)
                
                if view_name == 'top':
                    up = np.array([0, 0, -1])
                else:  # bottom
                    up = np.array([0, 0, 1])
                
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward) / np.linalg.norm(np.cross(right, forward))
                
                camera_transform = np.eye(4)
                camera_transform[:3, 0] = right
                camera_transform[:3, 1] = up
                camera_transform[:3, 2] = -forward
                camera_transform[:3, 3] = camera_pos
                
                c2w = camera_transform.copy()
                camera_params[view_name] = {
                    'intrinsics': intrinsic.tolist(),
                    'c2w': c2w.tolist(),
                    'resolution': [resolution, resolution]
                }
                
                # Render 1: Clean image (for SAM - no markers)
                scene_clean = trimesh.Scene([self.mesh])
                scene_clean.camera_transform = camera_transform
                scene_clean.camera.fov = [50, 50]
                
                png_clean = scene_clean.save_image(resolution=[resolution, resolution], visible=True)
                image_clean = np.array(Image.open(trimesh.util.wrap_as_stream(png_clean)))[:, :, :3]
                
                rendered_images[view_name] = image_clean
                
                output_path_clean = os.path.join(self.output_dir, f"view_{view_name}.png")
                Image.fromarray(image_clean).save(output_path_clean)
                
                # Render 2: Marked image (for GPT context - with colored markers)
                scene_marked = trimesh.Scene([self.mesh])
                
                for i, point in enumerate(self.selected_points):
                    sphere = trimesh.creation.icosphere(subdivisions=3, radius=self.size * marker_size)
                    sphere.apply_translation(point)
                    
                    color_rgb = np.array(COLORS[i]) * 255
                    color_rgba = np.array([color_rgb[0], color_rgb[1], color_rgb[2], 255], dtype=np.uint8)
                    sphere.visual.face_colors = np.tile(color_rgba, (len(sphere.faces), 1))
                    
                    scene_marked.add_geometry(sphere)
                
                scene_marked.camera_transform = camera_transform
                scene_marked.camera.fov = [50, 50]
                
                png_marked = scene_marked.save_image(resolution=[resolution, resolution], visible=True)
                image_marked = np.array(Image.open(trimesh.util.wrap_as_stream(png_marked)))[:, :, :3]
                
                rendered_images_marked[view_name] = image_marked
                
                output_path_marked = os.path.join(self.output_dir, f"view_{view_name}_marked.png")
                Image.fromarray(image_marked).save(output_path_marked)
                
                print(f"  ✓ {view_name:8s} (clean + marked)")
                
            except Exception as e:
                print(f"  ✗ {view_name:8s}: Failed - {e}")
        
        self.rendered_views = rendered_images  # Clean for SAM
        self.rendered_views_marked = rendered_images_marked  # Marked for GPT
        self.camera_params = camera_params
        
        # Save camera parameters
        cam_path = os.path.join(self.output_dir, "camera_params.json")
        with open(cam_path, 'w') as f:
            json.dump(camera_params, f, indent=2)
        
        print(f"\n  ✓ Clean images (for SAM): view_{{top,bottom}}.png")
        print(f"  ✓ Marked images (for GPT): view_{{top,bottom}}_marked.png")
        
        return rendered_images
    
    def segment_all_views_with_sam(self):
        """Run SAM on top and bottom views to get candidates"""
        print("\n" + "="*70)
        print("SAM Segmentation on Top and Bottom Views")
        print("="*70)
        
        self.all_candidates = {}
        
        for view_name, image in self.rendered_views.items():
            print(f"\n[{view_name.upper()}]")
            print(f"  Generating SAM masks...", end='')
            
            outputs = self.sam_generator.generate(image)
            print(f" {len(outputs)} masks")
            
            if len(outputs) == 0:
                print(f"  ⚠ No masks generated")
                continue
            
            # Filter by size
            H, W = image.shape[:2]
            valid = [o for o in outputs if 0.01 < o['area']/(H*W) < 0.7]
            print(f"  Filtered to {len(valid)} valid masks")
            
            self.all_candidates[view_name] = valid
            
            # Visualize all candidates
            self._visualize_all_candidates(image, view_name, valid)
        
        print(f"\n✓ Generated candidates for {len(self.all_candidates)} views")
        return self.all_candidates
    
    def _visualize_all_candidates(self, image: np.ndarray, view_name: str, outputs: List):
        """Visualize all SAM candidates"""
        n_candidates = len(outputs)
        
        if n_candidates == 0:
            return
        
        # Arrange in grid
        n_cols = min(5, n_candidates)
        n_rows = (n_candidates + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        fig.suptitle(f'{view_name.upper()} - All SAM Candidates', fontsize=16, fontweight='bold')
        
        for i in range(len(axes)):
            if i < n_candidates:
                mask = outputs[i]['segmentation']
                
                vis = image.copy()
                vis[mask > 0] = vis[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                
                axes[i].imshow(vis.astype(np.uint8))
                axes[i].set_title(f'#{i+1}', fontsize=14, fontweight='bold')
                axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.debug_dir, f"all_candidates_{view_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Candidates image: {save_path}")
    
    def select_parts_manual(self) -> Dict[str, np.ndarray]:
        """Manual selection: select one part, rest is automatically the other part"""
        print("\n" + "="*70)
        print("Manual Part Selection (Simplified)")
        print("="*70)
        print("\nPart names:")
        for i, (color, name) in enumerate(zip(COLOR_NAMES, self.part_names)):
            print(f"  • {color} marker → {name.upper()}")
        
        # Select which view to use
        print("\nAvailable views:")
        available_views = list(self.all_candidates.keys())
        for i, view in enumerate(available_views, 1):
            print(f"  {i}. {view.upper()}")
        
        while True:
            choice = input(f"\nSelect view (1-{len(available_views)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(available_views):
                selected_view = available_views[int(choice) - 1]
                break
            else:
                print(f"✗ Invalid, enter 1-{len(available_views)}")
        
        print(f"\n✓ Using {selected_view.upper()} view")
        print(f"Check: debug/all_candidates_{selected_view}.png")
        print(f"Check: view_{selected_view}_marked.png")
        
        # Only select first part, rest is automatically second part
        first_part = self.part_names[0]
        second_part = self.part_names[1]
        first_color = COLOR_NAMES[0]
        
        print(f"\n{'='*70}")
        print(f"Select mask for {first_part.upper()} ({first_color} marker)")
        print(f"Everything else will automatically be {second_part.upper()}")
        print(f"{'='*70}")
        
        n_candidates = len(self.all_candidates[selected_view])
        
        while True:
            choice = input(f"\nSelect {first_part} mask (1-{n_candidates}): ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= n_candidates:
                idx = int(choice) - 1
                first_mask = self.all_candidates[selected_view][idx]['segmentation']
                print(f"✓ Selected #{choice} for {first_part}")
                break
            else:
                print(f"✗ Invalid, enter 1-{n_candidates}")
        
        # Dilate mask by 10 pixels
        print(f"  Dilating mask by 10 pixels...")
        kernel = np.ones((10, 10), np.uint8)
        first_mask_dilated = cv2.dilate(first_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        
        # Create inverted mask for second part
        second_mask = ~first_mask_dilated
        
        selected_masks = {
            f"{first_part}_{selected_view}": first_mask,
            f"{second_part}_{selected_view}": second_mask
        }
        
        print(f"\n✓ {first_part}: selected mask #{idx+1}")
        print(f"✓ {second_part}: inverted mask (everything else)")
        
        return selected_masks
    
    def select_parts_with_gpt(self) -> Dict[str, np.ndarray]:
        """GPT automatic selection: choose view, select one part, rest is the other"""
        print("\n" + "="*70)
        print("GPT Automatic Part Selection (Fully Automatic)")
        print("="*70)
        
        if not HAS_OPENAI:
            print("  ✗ OpenAI not available, use manual selection")
            return self.select_parts_manual()
        
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("  ✗ OPENAI_API_KEY not set, use manual selection")
            return self.select_parts_manual()
        
        openai.api_key = api_key
        
        print("\nGPT will:")
        print("  1. Analyze both views (top and bottom)")
        print("  2. Choose the best view for segmentation")
        print("  3. Select mask for first part")
        print("  4. Rest is automatically second part")
        
        # Ask GPT to choose the best view
        print("\nStep 1: GPT choosing best view...")
        best_view = self._gpt_choose_best_view()
        
        if not best_view:
            print("  ✗ GPT view selection failed, trying all views...")
            # Fallback: try both views
            for view_name in ['top', 'bottom']:
                if view_name in self.all_candidates:
                    best_view = view_name
                    break
        
        if not best_view:
            print("\n  ✗ No valid views, switching to manual")
            return self.select_parts_manual()
        
        print(f"  ✓ GPT chose: {best_view.upper()} view")
        
        # Ask GPT to select mask for first part
        print(f"\nStep 2: GPT selecting mask from {best_view}...")
        candidates_img_path = os.path.join(self.debug_dir, f"all_candidates_{best_view}.png")
        
        if not os.path.exists(candidates_img_path):
            print("\n  ✗ Candidates image not found, switching to manual")
            return self.select_parts_manual()
        
        first_part = self.part_names[0]
        first_color = COLOR_NAMES[0]
        
        best_mask_idx = self._gpt_select_part(
            candidates_img_path,
            first_part,
            first_color,
            best_view,
            len(self.all_candidates[best_view])
        )
        
        if best_mask_idx is None:
            print("\n  ✗ GPT mask selection failed, switching to manual")
            return self.select_parts_manual()
        
        print(f"  ✓ GPT selected mask #{best_mask_idx+1}")
        
        # Get the selected mask and dilate it
        first_part = self.part_names[0]
        second_part = self.part_names[1]
        
        first_mask = self.all_candidates[best_view][best_mask_idx]['segmentation']
        
        # Dilate mask by 10 pixels
        print(f"  Dilating mask by 10 pixels...")
        kernel = np.ones((10, 10), np.uint8)
        first_mask_dilated = cv2.dilate(first_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        
        # Create inverted mask for second part
        second_mask = ~first_mask_dilated
        
        selected_masks = {
            f"{first_part}_{best_view}": first_mask_dilated,
            f"{second_part}_{best_view}": second_mask
        }
        
        print(f"\n✓ View: {best_view}")
        print(f"✓ {first_part}: mask #{best_mask_idx+1} (dilated 10px)")
        print(f"✓ {second_part}: inverted (everything else)")
        
        return selected_masks
    
    def _gpt_choose_best_view(self) -> str:
        """Ask GPT to choose the best view for segmentation"""
        
        try:
            # Encode both top and bottom marked views
            view_images = []
            for vname in ['top', 'bottom']:
                marked_path = os.path.join(self.output_dir, f"view_{vname}_marked.png")
                if os.path.exists(marked_path):
                    with open(marked_path, 'rb') as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                        view_images.append({
                            "type": "text",
                            "text": f"\n--- {vname.upper()} VIEW ---"
                        })
                        view_images.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                                "detail": "high"
                            }
                        })
            
            if not view_images:
                return None
            
            part_names_str = " and ".join(self.part_names)
            
            prompt = f"""You are analyzing an articulated object to determine the best view for segmentation.

The object has been divided into {len(self.part_names)} parts with specific names:
- Part 1 (RED marker): "{self.part_names[0]}"
- Part 2 (GREEN marker): "{self.part_names[1]}"

CRITICAL REQUIREMENT: The "{self.part_names[0]}" part (RED) and the "{self.part_names[1]}" part (GREEN) must NOT overlap in the 2D projection.

TASK: Choose which view (TOP or BOTTOM) shows the parts with NO OVERLAP.

Evaluation criteria (in order of importance):
1. **No overlap**: The "{self.part_names[0]}" (RED) and "{self.part_names[1]}" (GREEN) must be clearly separated in 2D
2. Both parts visible: Both "{self.part_names[0]}" and "{self.part_names[1]}" can be seen
3. Clear boundaries: The separation line between the two parts is visible
4. Complete coverage: Both parts are fully visible (not cut off)

INSTRUCTIONS:
1. First, carefully examine the TOP view:
   - Find the RED marker → that's the "{self.part_names[0]}" part
   - Find the GREEN marker → that's the "{self.part_names[1]}" part
   - Do the "{self.part_names[0]}" and "{self.part_names[1]}" regions overlap in this 2D view?

2. Then, examine the BOTTOM view:
   - Find the RED marker → that's the "{self.part_names[0]}" part
   - Find the GREEN marker → that's the "{self.part_names[1]}" part
   - Do the "{self.part_names[0]}" and "{self.part_names[1]}" regions overlap in this 2D view?

3. Compare and choose the view where "{self.part_names[0]}" and "{self.part_names[1]}" have LESS or NO overlap

Return your answer in this format:
THINKING: [Analyze TOP view - describe where RED "{self.part_names[0]}" is, where GREEN "{self.part_names[1]}" is, do they overlap? Then analyze BOTTOM view - same analysis. Which has less overlap between these two specific parts?]
ANSWER: top or bottom

Example:
THINKING: In the TOP view, the "{self.part_names[0]}" (RED marker) is positioned directly above the "{self.part_names[1]}" (GREEN marker), causing significant overlap in 2D projection. In the BOTTOM view, the "{self.part_names[1]}" (GREEN) is clearly visible at the center, while the "{self.part_names[0]}" (RED) is at the edge, with minimal overlap. BOTTOM view provides better separation of these two specific parts.
ANSWER: bottom
"""

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *view_images
                ]
            }]
            
            response = openai.ChatCompletion.create(
                model=os.environ.get('OPENAI_GPT_MODEL', 'gpt-4-turbo'),
                messages=messages,
                max_tokens=300,  # Allow space for thinking
                temperature=0.3
            )
            
            result = response['choices'][0]['message']['content'].strip()
            
            # Print GPT's thinking
            if "THINKING:" in result:
                thinking_part = result.split("ANSWER:")[0].replace("THINKING:", "").strip()
                print(f"\n  GPT's analysis:")
                print(f"  {thinking_part}\n")
            
            # Extract answer
            if "ANSWER:" in result:
                answer = result.split("ANSWER:")[-1].strip().lower()
                answer = answer.split()[0] if answer.split() else answer  # Get first word
            else:
                answer = result.lower()
            
            if answer in ['top', 'bottom']:
                return answer
            else:
                print(f"  ⚠ GPT returned unexpected: {result}")
                return None
                
        except Exception as e:
            print(f"  ⚠ GPT view selection failed: {e}")
            return None
    
    def _gpt_select_part(self, candidates_img_path: str, part_name: str, color_name: str, view_name: str, n_candidates: int):
        """Ask GPT to select which candidate represents the part (with marked images as context)"""
        
        try:
            # Encode candidates image
            with open(candidates_img_path, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()
            
            # Encode all marked images for context
            context_images = []
            for vname in self.rendered_views_marked.keys():
                marked_path = os.path.join(self.output_dir, f"view_{vname}_marked.png")
                if os.path.exists(marked_path):
                    with open(marked_path, 'rb') as f:
                        marked_b64 = base64.b64encode(f.read()).decode()
                        context_images.append({
                            "type": "text",
                            "text": f"\n--- {vname.upper().replace('_', ' ')} VIEW (with colored markers) ---"
                        })
                        context_images.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{marked_b64}",
                                "detail": "low"
                            }
                        })
            
            prompt = f"""You are analyzing a 3D articulated object.

CONTEXT: The object is shown from 6 oblique diagonal views with colored markers.
The {color_name} colored marker was placed on the "{part_name}" part.

TASK: Identify which segmentation mask represents the "{part_name}" part (marked with {color_name}) in the {view_name.upper().replace('_', ' ')} view.

The final image shows {n_candidates} segmentation candidates, labeled #1, #2, #3, etc.
Each candidate shows a green overlay on the segmented region.

Your task:
1. First look at all 6 marked views to understand the 3D structure
2. Find where the {color_name} marker is located (that's the "{part_name}" part)
3. Examine the candidates for {view_name} view
4. Select which candidate best segments ONLY the "{part_name}" part

Return ONLY a single number:
- 0: The "{part_name}" part is NOT clearly visible in the {view_name} view
- 1-{n_candidates}: The number of the best candidate for "{part_name}"

Return only the number, nothing else.
"""

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *context_images,
                    {"type": "text", "text": f"\n--- CANDIDATES for {view_name.upper()} view ---"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}
                    }
                ]
            }]
            
            response = openai.ChatCompletion.create(
                model=os.environ.get('OPENAI_GPT_MODEL', 'gpt-4-turbo'),
                messages=messages,
                max_tokens=10,
                temperature=0.3
            )
            
            result = response['choices'][0]['message']['content'].strip()
            
            selected = int(result)
            if selected == 0:
                return None
            elif 1 <= selected <= n_candidates:
                return selected - 1
            else:
                print(f" ⚠ Invalid: {selected}")
                return None
                
        except Exception as e:
            print(f" ⚠ GPT failed: {e}")
            return None
    
    def split_mesh_by_masks(self, selected_masks: Dict[str, np.ndarray]) -> Dict[str, trimesh.Trimesh]:
        """
        Split mesh using single view with inverted mask
        
        Strategy:
        1. One mask for first part (selected)
        2. Inverted mask for second part (automatic)
        3. Project face centroids to 2D
        4. Assign based on mask
        5. Extract submeshes
        """
        print("\n" + "="*70)
        print("Splitting Mesh by Single View")
        print("="*70)
        
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        n_faces = len(faces)
        
        # Get the view and masks
        part_masks = {}
        view_name = None
        for key, mask_2d in selected_masks.items():
            part_name, vname = key.split('_', 1)
            part_masks[part_name] = mask_2d
            view_name = vname
        
        print(f"\nUsing {view_name} view for splitting...")
        print(f"Parts: {', '.join(self.part_names)}")
        
        # Get camera parameters
        cam_params = self.camera_params[view_name]
        intr = np.array(cam_params['intrinsics'])
        c2w = np.array(cam_params['c2w'])
        
        # Get image dimensions
        first_mask = list(part_masks.values())[0]
        H, W = first_mask.shape
        
        # Compute face centroids
        print("\n  Computing face centroids...")
        face_centroids = np.mean(vertices[faces], axis=1)  # (n_faces, 3)
        
        # Project centroids to 2D
        print("  Projecting to 2D...")
        uv = self._project_vertices_to_2d(face_centroids, intr, c2w)
        
        # Assign faces
        print("  Assigning faces to parts...")
        part_faces = {name: [] for name in self.part_names}
        
        in_bounds = 0
        out_of_bounds = 0
        
        for f_idx in range(n_faces):
            u, v = uv[f_idx]
            
            # Check if in bounds
            if not (0 <= u < W and 0 <= v < H):
                out_of_bounds += 1
                continue
            
            in_bounds += 1
            u_int, v_int = int(u), int(v)
            
            # Check which mask this face falls into
            for part_name, mask in part_masks.items():
                if mask[v_int, u_int]:
                    part_faces[part_name].append(f_idx)
                    break
        
        print(f"\n  Projection stats:")
        print(f"    In bounds: {in_bounds:,} faces")
        print(f"    Out of bounds: {out_of_bounds:,} faces")
        
        print(f"\n  Face assignment:")
        for part_name in self.part_names:
            n_faces_part = len(part_faces[part_name])
            print(f"    {part_name}: {n_faces_part:,} faces")
        
        total_assigned = sum(len(faces) for faces in part_faces.values())
        print(f"    Unassigned: {n_faces - total_assigned:,}")
        
        # Extract meshes by faces
        result_meshes = {}
        for part_name in self.part_names:
            if len(part_faces[part_name]) == 0:
                print(f"\n  ⚠ WARNING: {part_name} has 0 faces!")
                print(f"    This means the mask didn't match any face centroids")
                print(f"    Possible causes:")
                print(f"      - Wrong mask selected")
                print(f"      - Projection failed")
                print(f"      - Mask too small")
            
            mesh = self._extract_mesh_by_faces(part_faces[part_name])
            result_meshes[part_name] = mesh
            print(f"\n  {part_name}: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
            
            if len(mesh.vertices) == 0:
                raise ValueError(f"ERROR: {part_name} mesh is empty! Cannot continue with URDF generation.")
        
        return result_meshes
    
    def _project_vertices_to_2d(self, vertices: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
        """
        Project 3D vertices to 2D image coordinates
        
        NOTE: Trimesh uses OpenGL camera convention:
        - Camera looks down -Z axis
        - Y-up coordinate system
        - Image origin at bottom-left
        
        But masks use typical image convention:
        - Origin at top-left
        - Y-down
        """
        # c2w is camera-to-world transform from trimesh
        cam_pos = c2w[:3, 3]
        R_c2w = c2w[:3, :3]
        
        # World to camera: inverse of c2w
        # For rotation matrix: inverse = transpose
        R_w2c = R_c2w.T
        
        # Transform vertices to camera space
        vertices_world = vertices - cam_pos
        vertices_cam = (R_w2c @ vertices_world.T).T
        
        # In OpenGL/trimesh camera space:
        # +X = right, +Y = up, -Z = forward (camera looks down -Z)
        # We need points with negative Z (in front of camera)
        valid_depth = vertices_cam[:, 2] < 0  # Note: negative Z is forward!
        
        # Perspective projection
        # Flip Z for projection (make it positive for division)
        vertices_cam_proj = vertices_cam.copy()
        vertices_cam_proj[:, 2] = -vertices_cam_proj[:, 2]  # Now positive Z = depth
        
        # Project: [x, y, z] → [f*x/z, f*y/z]
        uv_homog = vertices_cam_proj @ K.T
        uv = uv_homog[:, :2] / (uv_homog[:, 2:3] + 1e-8)
        
        # Flip Y coordinate: OpenGL bottom-left origin → image top-left origin
        # Image height is in K[1,2] * 2
        image_height = K[1, 2] * 2
        uv[:, 1] = image_height - uv[:, 1]
        
        # Mark invalid projections (behind camera or invalid)
        uv[~valid_depth] = -999999
        
        return uv
    
    def _extract_mesh_by_faces(self, face_indices: List[int]) -> trimesh.Trimesh:
        """
        Extract submesh by selected faces
        
        Args:
            face_indices: List of face indices to extract
        
        Returns:
            New trimesh with only selected faces and their vertices
        """
        if len(face_indices) == 0:
            # Return empty mesh
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
        
        # Get selected faces
        selected_faces = self.mesh.faces[face_indices]
        
        # Find all vertices used by these faces
        used_vertices = np.unique(selected_faces.flatten())
        
        # Create vertex mapping
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        
        # Extract vertices
        new_vertices = self.mesh.vertices[used_vertices]
        
        # Remap faces to new vertex indices
        new_faces = np.array([[old_to_new[v] for v in face] for face in selected_faces])
        
        return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    
    def run_interactive(self) -> Dict[str, trimesh.Trimesh]:
        """
        Run complete pipeline
        
        Returns:
            Dict with 'lid' and 'body' meshes
        """
        print("\n" + "="*70)
        print("Simplified SAM Pipeline")
        print("="*70)
        print("\nStages:")
        print("  1. Select points to mark parts")
        print("  2. Name each part (manual or GPT)")
        print("  3. Render top & bottom views")
        print("  4. SAM generates candidates")
        print("  5. Select one part mask, rest is automatic")
        print("  6. Split mesh")
        print("="*70)
        
        # Stage 1: Point selection
        print("\n[Stage 1/6] Point Selection")
        print("-" * 70)
        self.select_points(max_points=2)
        
        # Stage 2: Name parts
        print("\n[Stage 2/6] Name Parts")
        print("-" * 70)
        self.input_part_names()
        self.visualize_selected_points()
        
        # Stage 3: Render
        print("\n[Stage 3/6] Rendering Top & Bottom Views")
        print("-" * 70)
        self.render_views()
        
        # Stage 4: SAM on all views
        print("\n[Stage 4/6] SAM Segmentation")
        print("-" * 70)
        self.segment_all_views_with_sam()
        
        # Stage 5: Part selection (simplified)
        print("\n[Stage 5/6] Part Selection (Simplified)")
        print("-" * 70)
        print("\nChoose selection method:")
        print("  1. Manual selection (you choose one mask) ")
        print("  2. GPT automatic selection (AI chooses)")
        print("\nNote: You only select ONE mask for the first part")
        print("      Everything else is automatically the second part")
        
        choice = input("\nSelect (1/2, default=1): ").strip()
        
        if choice == '2':
            print("\nUsing GPT for automatic selection...")
            selected_masks = self.select_parts_with_gpt()
        else:
            print("\nUsing manual selection...")
            selected_masks = self.select_parts_manual()
        
        if not selected_masks:
            raise ValueError("No masks selected")
        
        # Stage 6: Split mesh
        print("\n[Stage 6/6] Splitting Mesh")
        print("-" * 70)
        parts = self.split_mesh_by_masks(selected_masks)
        
        print("\n" + "="*70)
        print("Pipeline Complete!")
        print("="*70)
        
        return parts
    
    def export_parts(self, parts: Dict[str, trimesh.Trimesh], output_dir: str):
        """Export parts to files"""
        print(f"\nExporting parts...")
        
        exported_files = {}
        for part_name, mesh in parts.items():
            output_path = os.path.join(output_dir, f"{part_name}.glb")
            save_mesh(mesh, output_path)
            exported_files[part_name] = output_path
            print(f"  {part_name}: {output_path}")
        
        return exported_files

