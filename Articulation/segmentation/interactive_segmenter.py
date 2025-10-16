#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete SAM+CLIP Interactive Segmentation Pipeline
Point Selection → Rendering → CLIP Prompts → SAM+CLIP Segmentation → 3D Projection
"""

import os
import sys
import json
import base64
import io
import numpy as np
import trimesh
import open3d as o3d
from PIL import Image
import cv2
import torch
import clip
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
import openai

# Add FGVP path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'fgvp-reclip'))
try:
    from fine_grained_visual_prompt import FGVP_ENSEMBLE
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    HAS_FGVP = True
except:
    HAS_FGVP = False
    print("Warning: FGVP not available, using basic CLIP")

from autoseg.utils.point_utils import project_pcd, get_depth_map, mask_pcd_2d
from autoseg.utils.mesh_utils import load_mesh, save_mesh

# Color scheme
COLORS = [
    [1.0, 0.0, 0.0],  # Red
    [0.0, 1.0, 0.0],  # Green
    [0.0, 0.0, 1.0],  # Blue
    [1.0, 1.0, 0.0],  # Yellow
    [1.0, 0.0, 1.0],  # Magenta
    [0.0, 1.0, 1.0],  # Cyan
    [1.0, 0.5, 0.0],  # Orange
    [0.5, 0.0, 1.0],  # Purple
]

COLOR_NAMES = [
    "RED", "GREEN", "BLUE", "YELLOW",
    "MAGENTA", "CYAN", "ORANGE", "PURPLE"
]


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


class InteractiveSegmenter:
    """Complete SAM+CLIP Interactive Segmentation Pipeline"""
    
    def __init__(self, glb_path: str, sam_checkpoint: str, output_dir: str):
        print("\n" + "="*70)
        print("SAM+CLIP Interactive Segmentation Pipeline")
        print("="*70)
        
        self.glb_path = glb_path
        self.sam_checkpoint = sam_checkpoint
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Debug directory
        self.debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        self.selected_points = []
        self.selected_indices = []
        
        self._load_model()
        self._init_models()
        
        print("✓ Initialization complete\n")
    
    def _load_model(self):
        """Load 3D model"""
        print(f"\nLoading model: {self.glb_path}")
        
        self.mesh = load_mesh(self.glb_path)
        
        # Convert to Open3D
        self.o3d_mesh = o3d.geometry.TriangleMesh()
        self.o3d_mesh.vertices = o3d.utility.Vector3dVector(self.mesh.vertices)
        self.o3d_mesh.triangles = o3d.utility.Vector3iVector(self.mesh.faces)
        self.o3d_mesh.compute_vertex_normals()
        
        # Compute bounding box
        self.bounds = self.mesh.bounds
        self.center = (self.bounds[0] + self.bounds[1]) / 2
        self.size = np.linalg.norm(self.bounds[1] - self.bounds[0])
        
        print(f"  Vertices: {len(self.mesh.vertices):,}")
        print(f"  Faces: {len(self.mesh.faces):,}")
        print(f"  Size: {self.size:.3f}")
        
        # Sample point cloud for 3D operations
        self.pcd = self.o3d_mesh.sample_points_uniformly(number_of_points=100000)
        self.points_3d = np.asarray(self.pcd.points)
        print(f"  Sampled point cloud: {len(self.points_3d):,} points")
    
    def _init_models(self):
        """Initialize SAM and CLIP models"""
        print("\nLoading AI models...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {self.device}")
        
        # SAM
        print("  Loading SAM...")
        self.sam_model = sam_model_registry['vit_h'](checkpoint=self.sam_checkpoint).to(self.device)
        self.sam_generator = SamAutomaticMaskGenerator(
            self.sam_model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.75,
            min_mask_region_area=100,
        )
        
        # CLIP
        print("  Loading CLIP...")
        encoder, _ = clip.load("ViT-L/14@336px", device=self.device)
        self.clip_model = ClipModel(encoder, clip.tokenize, self.device)
        
        # FGVP (if available)
        if HAS_FGVP:
            print("  Initializing FGVP...")
            self.fgvp = FGVP_ENSEMBLE(
                color_line='red', thickness=2, color_mask='green', alpha=0.5,
                clip_processing='resize', clip_image_size=336,
                resize_transform_clip=ResizeLongestSide(336),
                pixel_mean=torch.tensor(IMAGENET_DEFAULT_MEAN).view(-1, 1, 1).to(self.device) * 255,
                pixel_std=torch.tensor(IMAGENET_DEFAULT_STD).view(-1, 1, 1).to(self.device) * 255,
                blur_std_dev=100, mask_threshold=self.sam_model.mask_threshold,
                contour_scale=1.0, device=self.device
            )
        
        print("  ✓ Models loaded\n")
    
    def start_interactive_selection(self, max_points: int = 5):
        """Start interactive point selection (point cloud mode)"""
        print("\n" + "="*70)
        print("Interactive Point Selection")
        print("="*70)
        print("Instructions:")
        print("  • [Shift + Left Click] - Select point")
        print("  • [Shift + Right Click] - Remove last point")
        print("  • [Q] or close window - Finish selection")
        print(f"  • Max points: {max_points}")
        print("  • Rotate/zoom to view different angles")
        print("="*70)
        
        # Create point cloud for display and selection
        display_pcd = self.o3d_mesh.sample_points_uniformly(number_of_points=50000)
        display_pcd.paint_uniform_color([0.7, 0.7, 0.7])
        
        # Create visualization window
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(
            window_name="Point Selection - [Shift + Click] on points",
            width=1280, height=720
        )
        vis.add_geometry(display_pcd)
        
        # Render options
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 5.0
        
        print("\nTips:")
        print("  • Zoom in to see points clearly")
        print("  • Click ON points, not on background\n")
        
        vis.run()
        picked_points = vis.get_picked_points()
        vis.destroy_window()
        
        if len(picked_points) == 0:
            print("\n⚠ No points selected")
            print("Tip: Hold Shift and click ON points (not background)")
            return False
        
        if len(picked_points) > max_points:
            print(f"\n⚠ Selected {len(picked_points)} points, using first {max_points}")
            picked_points = picked_points[:max_points]
        
        # Map point cloud indices to mesh vertices
        pcd_points = np.asarray(display_pcd.points)
        mesh_vertices = np.asarray(self.o3d_mesh.vertices)
        
        for idx in picked_points:
            point = pcd_points[idx]
            distances = np.linalg.norm(mesh_vertices - point, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_point = mesh_vertices[nearest_idx]
            
            self.selected_points.append(nearest_point)
            self.selected_indices.append(int(nearest_idx))
        
        print(f"\n✓ Successfully selected {len(self.selected_points)} points:")
        for i, (point, idx) in enumerate(zip(self.selected_points, self.selected_indices)):
            color_name = COLOR_NAMES[i % len(COLOR_NAMES)]
            print(f"  {i+1}. [{color_name:8s}] Position: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
        
        return True
    
    def visualize_selected_points(self):
        """Visualize selected points with colored spheres"""
        print("\nShowing selected points...")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Confirm Selected Points", width=1024, height=768)
        vis.add_geometry(self.o3d_mesh)
        
        # Add colored spheres for each point
        for i, point in enumerate(self.selected_points):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.size * 0.02)
            sphere.translate(point)
            color = COLORS[i % len(COLORS)]
            sphere.paint_uniform_color(color)
            vis.add_geometry(sphere)
        
        vis.run()
        vis.destroy_window()
    
    def render_views_with_colored_markers(self, resolution: int = 800, marker_size: float = 0.03):
        """Render 6 views for comprehensive coverage (with and without markers)"""
        print(f"\nRendering 6 views...")
        print(f"  Resolution: {resolution}x{resolution}")
        print(f"  Marker size: {marker_size * self.size:.3f}")
        
        # Define 6 views: 2 oblique + 4 orthogonal
        distance = self.size * 2.0
        h = distance  # Vertical distance
        d = distance * 0.7  # Horizontal offset (45 degrees)
        
        view_configs = {
            'top': np.array([-d, h, 0]),        # Oblique: top-left (45deg)
            'bottom': np.array([d, -h, 0]),     # Oblique: bottom-right (45deg)
            'front': np.array([0, 0, distance]),     # Orthogonal: front
            'back': np.array([0, 0, -distance]),    # Orthogonal: back
            'left': np.array([-distance, 0, 0]),    # Orthogonal: left
            'right': np.array([distance, 0, 0]),    # Orthogonal: right
        }
        
        rendered_images = {}  # Clean images for SAM
        rendered_images_marked = {}  # Marked images for display/GPT
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
                # Set up camera parameters
                camera_pos = self.center + camera_offset
                look_at = self.center
                forward = (look_at - camera_pos) / np.linalg.norm(look_at - camera_pos)
                
                if view_name == 'top':
                    up = np.array([0, 0, -1])
                elif view_name == 'bottom':
                    up = np.array([0, 0, 1])
                else:
                    up = np.array([0, 1, 0])
                
                right = np.cross(forward, up)
                if np.linalg.norm(right) < 1e-6:
                    up = np.array([0, 0, 1]) if view_name in ['top', 'bottom'] else np.array([1, 0, 0])
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
                
                # Render 1: Clean image (for SAM)
                scene_clean = trimesh.Scene()
                scene_clean.add_geometry(self.mesh.copy())
                scene_clean.camera_transform = camera_transform
                scene_clean.camera.fov = [50, 50]
                
                png_clean = scene_clean.save_image(resolution=[resolution, resolution], visible=True)
                image_clean = np.array(Image.open(trimesh.util.wrap_as_stream(png_clean)))[:, :, :3]
                
                rendered_images[view_name] = image_clean
                
                # Save clean image
                output_path_clean = os.path.join(self.output_dir, f"view_{view_name}.png")
                Image.fromarray(image_clean).save(output_path_clean)
                
                # Render 2: Marked image (for display and GPT context)
                scene_marked = trimesh.Scene()
                scene_marked.add_geometry(self.mesh.copy())
                
                # Add colored spheres
                for i, point in enumerate(self.selected_points):
                    sphere = trimesh.creation.icosphere(subdivisions=3, radius=self.size * marker_size)
                    sphere.apply_translation(point)
                    
                    color_rgb = np.array(COLORS[i % len(COLORS)]) * 255
                    color_rgba = np.array([color_rgb[0], color_rgb[1], color_rgb[2], 255], dtype=np.uint8)
                    sphere.visual.face_colors = np.tile(color_rgba, (len(sphere.faces), 1))
                    
                    scene_marked.add_geometry(sphere)
                
                scene_marked.camera_transform = camera_transform
                scene_marked.camera.fov = [50, 50]
                
                png_marked = scene_marked.save_image(resolution=[resolution, resolution], visible=True)
                image_marked = np.array(Image.open(trimesh.util.wrap_as_stream(png_marked)))[:, :, :3]
                
                rendered_images_marked[view_name] = image_marked
                
                # Save marked image
                output_path_marked = os.path.join(self.output_dir, f"view_{view_name}_marked.png")
                Image.fromarray(image_marked).save(output_path_marked)
                
                print(f"  ✓ {view_name:8s} (clean + marked)")
                
            except Exception as e:
                print(f"  ✗ {view_name:8s}: Failed - {e}")
        
        self.rendered_views = rendered_images  # Use clean images for SAM
        self.rendered_views_marked = rendered_images_marked  # Marked images for display
        self.camera_params = camera_params
        
        # Save camera parameters
        cam_path = os.path.join(self.output_dir, "camera_params.json")
        with open(cam_path, 'w') as f:
            json.dump(camera_params, f, indent=2)
        print(f"\n  ✓ Camera params saved")
        print(f"  ✓ Clean images (for SAM): view_{{top,bottom}}.png")
        print(f"  ✓ Marked images (for display): view_{{top,bottom}}_marked.png")
        
        return rendered_images
    
    def manual_input_clip_prompts(self) -> Dict[str, str]:
        """Manual input of CLIP text prompts"""
        print("\n" + "="*70)
        print("Input CLIP Text Prompts")
        print("="*70)
        print("Enter a text description (2-5 words) for each colored point")
        print("="*70)
        
        clip_prompts = {}
        
        for i, point in enumerate(self.selected_points):
            color_name = COLOR_NAMES[i % len(COLOR_NAMES)]
            
            print(f"\n[{color_name}] Point #{i+1}:")
            print(f"  Position: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
            print(f"  Hint: Check view_*_marked.png to see which part")
            print(f"  Examples: lid, body, edge, handle, top, bottom")
            
            while True:
                prompt = input(f"  Enter CLIP prompt: ").strip()
                if prompt:
                    clip_prompts[color_name] = prompt
                    print(f"  ✓ {color_name}: \"{prompt}\"")
                    break
                else:
                    print(f"  ✗ Cannot be empty")
        
        print(f"\n{'='*70}")
        print("Input complete! CLIP prompts:")
        for color, prompt in clip_prompts.items():
            print(f"  • {color:8s} → \"{prompt}\"")
        print(f"{'='*70}")
        
        return clip_prompts
    
    def query_gpt_for_clip_prompts(self, model_name: str = "unnamed_object") -> Dict[str, str]:
        """
        Use GPT Vision API to automatically generate CLIP text prompts
        
        Returns:
            dict: {color_name: CLIP prompt}
        """
        print(f"\nGPT Vision: Auto-generating CLIP prompts...")
        
        if not hasattr(self, 'rendered_views') or len(self.rendered_views) == 0:
            print("  ✗ Need to render views first")
            return {}
        
        # Check API key
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("  ✗ OPENAI_API_KEY environment variable not set")
            print("  Tip: export OPENAI_API_KEY=your_key")
            return {}
        
        openai.api_key = api_key
        
        # Encode images to base64
        print("  [1/3] Encoding images...")
        image_dict = {}
        for view_name, image in self.rendered_views.items():
            img_pil = Image.fromarray(image)
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            buffer.seek(0)
            b64_str = base64.b64encode(buffer.read()).decode()
            image_dict[view_name] = b64_str
        
        # Build prompt
        print("  [2/3] Building prompt...")
        
        color_info = "\n".join([
            f"  - {COLOR_NAMES[i]}: Point #{i+1} at position {self.selected_points[i]}"
            for i in range(len(self.selected_points))
        ])
        
        prompt = f"""You are an expert in analyzing 3D objects and identifying different parts.

I have a 3D articulated object with {len(self.selected_points)} colored markers placed on different parts. 
The object is shown from TOP and BOTTOM views.

Colored markers:
{color_info}

Your task:
1. Analyze the TOP and BOTTOM views to understand the overall structure
2. For EACH colored marker, identify what PART it represents
3. Generate a PART NAME (1-2 words) for each marker

IMPORTANT:
- Give the PART NAME like "lid", "cover"
- Use functional part names that describe the component itself
- Use simple, common English words
- Each name should be different to distinguish parts

Examples:
- "lid" (movable cover)
- "body" (main container)
- "cover" (protective top)
- "base" (bottom part)
- "handle" (grip part)
- "frame" (structural part)

Return ONLY a JSON object:
{{
  "RED": "part name",
  "GREEN": "part name"
}}

Only include the colors actually in the images. Use lowercase. Return ONLY the JSON.
"""

        # Call GPT Vision API
        print("  [3/3] Querying GPT...")
        
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high"
                        }
                    } for b64 in image_dict.values()]
                ]
            }]
            
            gpt_model = os.environ.get('OPENAI_GPT_MODEL', 'gpt-4-turbo')
            response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            gpt_output = response['choices'][0]['message']['content']
            print(f"\n  ✓ GPT response:")
            print("  " + "-"*66)
            print("  " + gpt_output.replace("\n", "\n  "))
            print("  " + "-"*66)
            
            # Parse JSON response
            try:
                import re
                json_match = re.search(r'\{.*\}', gpt_output, re.DOTALL)
                if json_match:
                    clip_prompts = json.loads(json_match.group())
                else:
                    clip_prompts = json.loads(gpt_output)
                
                print(f"\n  ✓ Successfully parsed CLIP prompts:")
                for color, prompt in clip_prompts.items():
                    print(f"     • {color:8s} → \"{prompt}\"")
                
                return clip_prompts
                
            except json.JSONDecodeError as e:
                print(f"\n  ⚠ JSON parsing failed: {e}")
                print(f"  Raw output: {gpt_output}")
                return {}
                
        except Exception as e:
            print(f"\n  ✗ GPT API call failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def save_intermediate_results(self, clip_prompts: Dict[str, str]):
        """Save intermediate results (for caching)"""
        print(f"\nSaving intermediate results...")
        
        # Save point info
        points_data = {
            'num_points': len(self.selected_points),
            'points': [
                {
                    'id': i,
                    'color': COLOR_NAMES[i % len(COLOR_NAMES)],
                    'color_rgb': COLORS[i % len(COLORS)],
                    'position': self.selected_points[i].tolist(),
                    'vertex_index': self.selected_indices[i],
                    'clip_prompt': clip_prompts.get(COLOR_NAMES[i % len(COLOR_NAMES)], "")
                }
                for i in range(len(self.selected_points))
            ]
        }
        
        points_file = os.path.join(self.output_dir, "selected_points.json")
        with open(points_file, 'w') as f:
            json.dump(points_data, f, indent=2)
        print(f"  ✓ Points: {points_file}")
        
        # Save CLIP prompts
        prompts_file = os.path.join(self.output_dir, "clip_prompts.json")
        with open(prompts_file, 'w') as f:
            json.dump(clip_prompts, f, indent=2)
        print(f"  ✓ CLIP prompts: {prompts_file}")
        
        return clip_prompts
    
    def segment_with_gpt_selection(self, clip_prompts: Dict[str, str]) -> Dict:
        """Segment parts with GPT Vision automatic candidate selection"""
        print("\n" + "="*70)
        print("SAM+CLIP Segmentation (GPT Automatic Selection)")
        print("="*70)
        
        # Check API key
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("  ✗ OPENAI_API_KEY not set, falling back to manual selection")
            return self.segment_with_manual_selection(clip_prompts)
        
        openai.api_key = api_key
        
        all_parts_results = {}
        
        for i, point in enumerate(self.selected_points):
            color_name = COLOR_NAMES[i % len(COLOR_NAMES)]
            text_prompt = clip_prompts.get(color_name, f"part {i+1}")
            
            print(f"\n{'='*70}")
            print(f"Segmenting part #{i+1}: {color_name} - \"{text_prompt}\"")
            print(f"{'='*70}")
            
            part_masks = {}
            
            for view_name, image in self.rendered_views.items():
                existing_mask = os.path.join(self.output_dir, f"mask_{color_name}_{view_name}.npy")
                
                if os.path.exists(existing_mask):
                    print(f"\n  [{view_name.upper()}]")
                    print(f"    ✓ Existing result found, loading...")
                    loaded_mask = np.load(existing_mask)
                    part_masks[view_name] = loaded_mask
                    continue
                
                print(f"\n  [{view_name.upper()}]")
                
                # SAM segmentation
                print(f"    [1/4] SAM generating candidates...", end='')
                outputs = self.sam_generator.generate(image)
                if len(outputs) == 0:
                    print(" No candidates")
                    continue
                print(f" {len(outputs)} masks")
                
                H, W = image.shape[:2]
                valid = [o for o in outputs if o['area']/(H*W) < 0.7]
                if len(valid) == 0:
                    print(f"    [2/4] No valid regions")
                    continue
                print(f"    [2/4] Kept {len(valid)} valid regions")
                
                # CLIP ranking
                print(f"    [3/4] CLIP ranking all candidates...")
                masks = torch.from_numpy(np.stack([x['segmentation'] for x in valid])).unsqueeze(1).to(self.device)
                boxes = torch.tensor(np.stack([x['bbox'] for x in valid])).float().to(self.device)
                boxes[:, 2] += boxes[:, 0]
                boxes[:, 3] += boxes[:, 1]
                centers = torch.stack([boxes[:, 0::2].mean(1), boxes[:, 1::2].mean(1)], 1)
                
                if HAS_FGVP:
                    clip_inputs = self.fgvp("blur_mask", image, centers, boxes, masks)
                else:
                    clip_inputs = []
                    for mask in masks:
                        masked = image.copy()
                        mask_np = mask.squeeze().cpu().numpy() > 0.5
                        masked[~mask_np] = 0
                        clip_inputs.append(torch.from_numpy(masked).to(self.device))
                    clip_inputs = torch.stack(clip_inputs)
                
                logits = self.clip_model(clip_inputs, [f"a photo of {text_prompt}"])
                scores = logits.squeeze().cpu().numpy()
                
                # Show all candidates sorted by CLIP score
                all_indices = np.argsort(scores)[::-1]  # All candidates, sorted by score
                n_total = len(all_indices)
                
                # Visualize candidates
                candidates_path = self._show_candidates(
                    image, view_name, valid, all_indices, scores, color_name, text_prompt
                )
                
                # GPT Vision selection
                print(f"    [4/4] GPT selecting best candidate...")
                selected_idx = self._gpt_select_candidate(
                    candidates_path, image, valid, all_indices, text_prompt, view_name, color_name
                )
                
                if selected_idx is None:
                    print(f"        ⊗ GPT skipped this view")
                    continue
                
                final_idx = all_indices[selected_idx]
                final_mask = masks[final_idx].squeeze().cpu().numpy()
                
                print(f"        ✓ GPT selected: #{selected_idx+1} (score {scores[final_idx]:.3f})")
                
                part_masks[view_name] = final_mask
                
                # Save results
                vis = image.copy()
                vis[final_mask > 0] = vis[final_mask > 0] * 0.5 + np.array(COLORS[i % len(COLORS)]) * 255 * 0.5
                
                cv2.imwrite(
                    os.path.join(self.output_dir, f"segment_{color_name}_{view_name}.png"),
                    cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
                )
                
                np.save(os.path.join(self.output_dir, f"mask_{color_name}_{view_name}.npy"), final_mask)
                print(f"        ✓ Saved")
            
            all_parts_results[color_name] = {
                'text_prompt': text_prompt,
                'point_index': i,
                'masks_2d': part_masks
            }
            
            print(f"\n  ✓ [{color_name}] Selected {len(part_masks)} views")
        
        self.segmentation_results = all_parts_results
        return all_parts_results
    
    def _gpt_select_part_to_segment(self, clip_prompts: Dict[str, str]) -> str:
        """Ask GPT which part should be segmented (usually the smaller/movable one)"""
        
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return None
        
        openai.api_key = api_key
        
        try:
            parts_info = "\n".join([
                f"  - {color}: \"{name}\""
                for color, name in clip_prompts.items()
            ])
            
            prompt = f"""You are analyzing an articulated object that has been divided into parts:

{parts_info}

TASK: Choose which ONE part should be explicitly segmented.
The other part will automatically be "everything else" (inverted mask).

Guidelines:
- Choose the SMALLER, MOVABLE part (e.g., lid, cover, door, flap)
- The larger, FIXED part (e.g., body, base, container) will be automatic

For example:
- If parts are "lid" and "body" → segment "lid" (smaller/movable)
- If parts are "door" and "frame" → segment "door" (movable)
- If parts are "cover" and "base" → segment "cover" (movable)

Return ONLY the color name (RED, GREEN, etc.) of the part to segment.
Return only one word: the color name.
"""

            gpt_model = os.environ.get('OPENAI_GPT_MODEL', 'gpt-4-turbo')
            response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3
            )
            
            result = response['choices'][0]['message']['content'].strip().upper()
            
            if result in COLOR_NAMES:
                return result
            else:
                return None
                
        except Exception as e:
            print(f"  ⚠ GPT failed: {e}")
            return None
    
    def segment_single_part_manual(self, selected_color: str, clip_prompts: Dict[str, str]):
        """Segment only the selected part manually, others will be inverted"""
        
        text_prompt = clip_prompts[selected_color]
        view_name = list(self.rendered_views.keys())[0]
        image = self.rendered_views[view_name]
        
        print(f"\nSegmenting: {selected_color} - \"{text_prompt}\"")
        print(f"View: {view_name.upper()}")
        
        # SAM + CLIP
        print(f"\n  [1/3] SAM generating candidates...", end='')
        outputs = self.sam_generator.generate(image)
        print(f" {len(outputs)} masks")
        
        H, W = image.shape[:2]
        valid = [o for o in outputs if o['area']/(H*W) < 0.7]
        print(f"  [2/3] Kept {len(valid)} valid regions")
        
        # CLIP ranking
        print(f"  [3/3] CLIP ranking...")
        masks = torch.from_numpy(np.stack([x['segmentation'] for x in valid])).unsqueeze(1).to(self.device)
        boxes = torch.tensor(np.stack([x['bbox'] for x in valid])).float().to(self.device)
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        centers = torch.stack([boxes[:, 0::2].mean(1), boxes[:, 1::2].mean(1)], 1)
        
        if HAS_FGVP:
            clip_inputs = self.fgvp("blur_mask", image, centers, boxes, masks)
        else:
            clip_inputs = []
            for mask in masks:
                masked = image.copy()
                mask_np = mask.squeeze().cpu().numpy() > 0.5
                masked[~mask_np] = 0
                clip_inputs.append(torch.from_numpy(masked).to(self.device))
            clip_inputs = torch.stack(clip_inputs)
        
        logits = self.clip_model(clip_inputs, [f"a photo of {text_prompt}"])
        scores = logits.squeeze().cpu().numpy()
        all_indices = np.argsort(scores)[::-1]
        
        # Show candidates
        candidates_path = self._show_candidates(
            image, view_name, valid, all_indices, scores, selected_color, text_prompt
        )
        
        # Manual selection
        print(f"\n  Check: {candidates_path}")
        print(f"  Total: {len(all_indices)} candidates")
        print(f"  CLIP recommends: #1")
        
        while True:
            choice = input(f"\n  Select candidate (1-{len(all_indices)}, Enter=1): ").strip()
            if choice == '' or choice == '1':
                selected_idx = 0
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(all_indices):
                selected_idx = int(choice) - 1
                break
            else:
                print(f"  ✗ Invalid, enter 1-{len(all_indices)}")
        
        # Get mask and dilate it
        final_idx = all_indices[selected_idx]
        selected_mask = masks[final_idx].squeeze().cpu().numpy()
        
        # Dilate mask by 10 pixels
        print(f"\n  Dilating mask by 10 pixels...")
        kernel = np.ones((10, 10), np.uint8)
        selected_mask_dilated = cv2.dilate(selected_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        
        # Save results for selected part
        self.segmentation_results = {}
        self.segmentation_results[selected_color] = {
            'text_prompt': text_prompt,
            'masks_2d': {view_name: selected_mask_dilated}
        }
        
        # Save mask to cache
        mask_file = os.path.join(self.output_dir, f"mask_{selected_color}_{view_name}.npy")
        np.save(mask_file, selected_mask_dilated)
        
        # Create inverted masks for other parts
        for color, name in clip_prompts.items():
            if color != selected_color:
                inverted_mask = ~selected_mask_dilated
                self.segmentation_results[color] = {
                    'text_prompt': name,
                    'masks_2d': {view_name: inverted_mask}
                }
                # Save inverted mask to cache
                mask_file = os.path.join(self.output_dir, f"mask_{color}_{view_name}.npy")
                np.save(mask_file, inverted_mask)
        
        print(f"✓ {selected_color}: selected mask #{selected_idx+1} (dilated 10px)")
        print(f"✓ Other parts: inverted mask")
        print(f"✓ Masks saved to cache")
    
    def segment_single_part_gpt(self, selected_color: str, clip_prompts: Dict[str, str]):
        """Segment only the selected part with GPT, others will be inverted"""
        
        text_prompt = clip_prompts[selected_color]
        view_name = list(self.rendered_views.keys())[0]
        image = self.rendered_views[view_name]
        
        print(f"\nSegmenting: {selected_color} - \"{text_prompt}\"")
        print(f"View: {view_name.upper()}")
        
        # SAM + CLIP
        print(f"\n  [1/3] SAM generating candidates...", end='')
        outputs = self.sam_generator.generate(image)
        print(f" {len(outputs)} masks")
        
        H, W = image.shape[:2]
        valid = [o for o in outputs if o['area']/(H*W) < 0.7]
        print(f"  [2/3] Kept {len(valid)} valid regions")
        
        # CLIP ranking
        print(f"  [3/3] CLIP ranking...")
        masks = torch.from_numpy(np.stack([x['segmentation'] for x in valid])).unsqueeze(1).to(self.device)
        boxes = torch.tensor(np.stack([x['bbox'] for x in valid])).float().to(self.device)
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        centers = torch.stack([boxes[:, 0::2].mean(1), boxes[:, 1::2].mean(1)], 1)
        
        if HAS_FGVP:
            clip_inputs = self.fgvp("blur_mask", image, centers, boxes, masks)
        else:
            clip_inputs = []
            for mask in masks:
                masked = image.copy()
                mask_np = mask.squeeze().cpu().numpy() > 0.5
                masked[~mask_np] = 0
                clip_inputs.append(torch.from_numpy(masked).to(self.device))
            clip_inputs = torch.stack(clip_inputs)
        
        logits = self.clip_model(clip_inputs, [f"a photo of {text_prompt}"])
        scores = logits.squeeze().cpu().numpy()
        all_indices = np.argsort(scores)[::-1]
        
        # Show candidates
        candidates_path = self._show_candidates(
            image, view_name, valid, all_indices, scores, selected_color, text_prompt
        )
        
        # GPT selection
        print(f"\n  GPT selecting best candidate...")
        selected_idx = self._gpt_select_candidate(
            candidates_path, image, valid, all_indices, text_prompt, view_name, selected_color
        )
        
        if selected_idx is None:
            print("  ✗ GPT failed, using CLIP recommendation")
            selected_idx = 0
        
        # Get mask and dilate it
        final_idx = all_indices[selected_idx]
        selected_mask = masks[final_idx].squeeze().cpu().numpy()
        
        print(f"  ✓ GPT selected: #{selected_idx+1}")
        
        # Dilate mask by 10 pixels
        print(f"  Dilating mask by 10 pixels...")
        kernel = np.ones((10, 10), np.uint8)
        selected_mask_dilated = cv2.dilate(selected_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        
        # Save results for selected part
        self.segmentation_results = {}
        self.segmentation_results[selected_color] = {
            'text_prompt': text_prompt,
            'masks_2d': {view_name: selected_mask_dilated}
        }
        
        # Save mask to cache
        mask_file = os.path.join(self.output_dir, f"mask_{selected_color}_{view_name}.npy")
        np.save(mask_file, selected_mask_dilated)
        
        # Create inverted masks for other parts
        for color, name in clip_prompts.items():
            if color != selected_color:
                inverted_mask = ~selected_mask_dilated
                self.segmentation_results[color] = {
                    'text_prompt': name,
                    'masks_2d': {view_name: inverted_mask}
                }
                # Save inverted mask to cache
                mask_file = os.path.join(self.output_dir, f"mask_{color}_{view_name}.npy")
                np.save(mask_file, inverted_mask)
        
        print(f"\n✓ {selected_color}: GPT selected #{selected_idx+1} (dilated 10px)")
        print(f"✓ Other parts: inverted mask")
        print(f"✓ Masks saved to cache")
    
    def _gpt_select_best_view(self) -> str:
        """Ask GPT to select the best view for segmentation from all 6 views"""
        
        try:
            # Encode all marked views
            view_images = []
            for view_name in self.rendered_views_marked.keys():
                marked_path = os.path.join(self.output_dir, f"view_{view_name}_marked.png")
                if os.path.exists(marked_path):
                    with open(marked_path, 'rb') as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                        view_images.append({
                            "type": "text",
                            "text": f"\n--- {view_name.upper()} VIEW ---"
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
            
            prompt = f"""You are analyzing an articulated object to choose the best view for 2D segmentation.

Colored markers:
- RED marker: First part
- GREEN marker: Second part

You are shown 6 different views.

CRITICAL REQUIREMENT: Choose the view where RED and GREEN parts have MINIMAL or NO OVERLAP in 2D.

Evaluation criteria (priority order):
1. **No overlap**: RED and GREEN should NOT overlap in 2D projection
2. Both visible: Both parts can be seen
3. Clear boundary: Separation between parts is visible

INSTRUCTIONS:
1. Check each view for overlap between RED and GREEN regions
2. Choose the view with least overlap

Return format:
THINKING: [Brief analysis of which views have overlap, which don't]
ANSWER: view_name

where view_name is one of: top, bottom, front, back, left, right

Example:
THINKING: FRONT shows RED and GREEN side-by-side without overlap. TOP has overlap. FRONT is best.
ANSWER: front
"""

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *view_images
                ]
            }]
            
            gpt_model = os.environ.get('OPENAI_GPT_MODEL', 'gpt-4-turbo')
            response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )
            
            result = response['choices'][0]['message']['content'].strip()
            
            # Print thinking
            if "THINKING:" in result:
                thinking = result.split("ANSWER:")[0].replace("THINKING:", "").strip()
                print(f"\n  GPT's analysis: {thinking}\n")
            
            # Extract answer
            if "ANSWER:" in result:
                answer = result.split("ANSWER:")[-1].strip().lower()
                answer = answer.split()[0] if answer.split() else answer
            else:
                answer = result.lower()
            
            valid_views = ['top', 'bottom', 'front', 'back', 'left', 'right']
            if answer in valid_views:
                return answer
            else:
                return None
                
        except Exception as e:
            print(f"  ⚠ GPT failed: {e}")
            return None
    
    def _gpt_select_candidate(
        self,
        candidates_image_path: str,
        original_image: np.ndarray,
        outputs: List,
        candidate_indices: np.ndarray,
        text_prompt: str,
        view_name: str,
        color_name: str
    ) -> int:
        """Use GPT Vision to automatically select best candidate with multi-view context"""
        
        try:
            # Encode candidate image
            with open(candidates_image_path, 'rb') as f:
                candidates_b64 = base64.b64encode(f.read()).decode()
            
            # Encode all available views for context
            context_images = []
            for vname in ['top', 'bottom']:
                view_path = os.path.join(self.output_dir, f"view_{vname}_marked.png")
                if os.path.exists(view_path):
                    with open(view_path, 'rb') as f:
                        view_b64 = base64.b64encode(f.read()).decode()
                        context_images.append({
                            "type": "text",
                            "text": f"\n--- {vname.upper()} VIEW (with colored markers) ---"
                        })
                        context_images.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{view_b64}",
                                "detail": "low"  # Low detail for context views
                            }
                        })
            
            # Build prompt
            prompt = f"""You are an expert in 3D object understanding and image segmentation quality assessment.

CONTEXT: The object is shown from TOP and BOTTOM views with colored markers on different parts.
You are analyzing the {color_name} colored marker which represents: "{text_prompt}"

CURRENT TASK: Select the best segmentation candidate for the "{text_prompt}" part in the {view_name.upper()} view.

The final image shows {len(candidate_indices)} segmentation candidates (labeled #1 to #{len(candidate_indices)}).
Each candidate shows a green overlay indicating the segmented region.
Candidates are sorted by CLIP score (best first).

Your task:
1. First, understand the 3D structure from TOP and BOTTOM views
2. Identify where the {color_name} marker ({text_prompt}) is located
3. Look at the current {view_name} view candidates
4. Select which candidate best isolates ONLY the "{text_prompt}" part

Evaluation criteria:
- Complete coverage of the "{text_prompt}" part
- Accurate boundaries (not too loose, not too tight)
- No inclusion of other parts
- Handles occlusion correctly

Return ONLY a single number:
- 0: The "{text_prompt}" part is NOT visible in the {view_name} view (occluded or facing away)
- 1-{len(candidate_indices)}: The number of the best candidate

Return only the number, nothing else.
"""

            # Build message with context images + candidate image
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *context_images,
                    {"type": "text", "text": f"\n--- CANDIDATES for {view_name.upper()} view ---"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{candidates_b64}",
                            "detail": "high"  # High detail for candidates
                        }
                    }
                ]
            }]
            
            gpt_model = os.environ.get('OPENAI_GPT_MODEL', 'gpt-4-turbo')
            response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=messages,
                max_tokens=10,
                temperature=0.3
            )
            
            gpt_output = response['choices'][0]['message']['content'].strip()
            
            # Parse number
            try:
                selected = int(gpt_output)
                if selected == 0:
                    return None
                elif 1 <= selected <= len(candidate_indices):
                    return selected - 1
                else:
                    print(f"        ⚠ GPT returned invalid number: {selected}, using CLIP recommendation")
                    return 0
            except ValueError:
                print(f"        ⚠ GPT returned non-number: {gpt_output}, using CLIP recommendation")
                return 0
                
        except Exception as e:
            print(f"        ⚠ GPT call failed: {e}, using CLIP recommendation")
            return 0
    
    def segment_with_manual_selection(self, clip_prompts: Dict[str, str]) -> Dict:
        """Segment parts with manual candidate selection"""
        print("\n" + "="*70)
        print("SAM+CLIP Segmentation (Manual Selection)")
        print("="*70)
        
        all_parts_results = {}
        
        for i, point in enumerate(self.selected_points):
            color_name = COLOR_NAMES[i % len(COLOR_NAMES)]
            text_prompt = clip_prompts.get(color_name, f"part {i+1}")
            
            print(f"\n{'='*70}")
            print(f"Segmenting part #{i+1}: {color_name} - \"{text_prompt}\"")
            print(f"{'='*70}")
            
            part_masks = {}
            
            for view_name, image in self.rendered_views.items():
                # Check for existing results (resume support)
                existing_mask = os.path.join(self.output_dir, f"mask_{color_name}_{view_name}.npy")
                
                if os.path.exists(existing_mask):
                    print(f"\n  [{view_name.upper()}]")
                    print(f"    ✓ Existing result found, loading...")
                    loaded_mask = np.load(existing_mask)
                    part_masks[view_name] = loaded_mask
                    continue
                
                print(f"\n  [{view_name.upper()}]")
                
                # SAM segmentation
                print(f"    [1/4] SAM generating candidates...", end='')
                outputs = self.sam_generator.generate(image)
                if len(outputs) == 0:
                    print(" No candidates")
                    continue
                print(f" {len(outputs)} masks")
                
                # Filter
                H, W = image.shape[:2]
                valid = [o for o in outputs if o['area']/(H*W) < 0.7]
                if len(valid) == 0:
                    print(f"    [2/4] No valid regions after filtering")
                    continue
                print(f"    [2/4] Kept {len(valid)} valid regions")
                
                # CLIP ranking
                print(f"    [3/4] CLIP ranking all candidates...")
                masks = torch.from_numpy(np.stack([x['segmentation'] for x in valid])).unsqueeze(1).to(self.device)
                boxes = torch.tensor(np.stack([x['bbox'] for x in valid])).float().to(self.device)
                boxes[:, 2] += boxes[:, 0]
                boxes[:, 3] += boxes[:, 1]
                centers = torch.stack([boxes[:, 0::2].mean(1), boxes[:, 1::2].mean(1)], 1)
                
                # Apply FGVP if available
                if HAS_FGVP:
                    clip_inputs = self.fgvp("blur_mask", image, centers, boxes, masks)
                else:
                    # Simple masking
                    clip_inputs = []
                    for mask in masks:
                        masked = image.copy()
                        mask_np = mask.squeeze().cpu().numpy() > 0.5
                        masked[~mask_np] = 0
                        clip_inputs.append(torch.from_numpy(masked).to(self.device))
                    clip_inputs = torch.stack(clip_inputs)
                
                logits = self.clip_model(clip_inputs, [f"a photo of {text_prompt}"])
                scores = logits.squeeze().cpu().numpy()
                
                # Show all candidates sorted by CLIP score
                all_indices = np.argsort(scores)[::-1]  # All candidates, sorted descending
                n_total = len(all_indices)
                
                # Visualize candidates
                candidates_path = self._show_candidates(
                    image, view_name, valid, all_indices, scores, color_name, text_prompt
                )
                
                # Manual selection
                print(f"    [4/4] Manual selection...")
                print(f"        → Candidates: {candidates_path}")
                print(f"        → Total candidates: {n_total}")
                print(f"        → CLIP recommends: #1")
                
                while True:
                    choice = input(f"        Select candidate (0=skip, 1-{n_total}, Enter=1): ").strip()
                    
                    if choice == '0':
                        print(f"        ⊗ Skipping view (occluded)")
                        selected_idx = None
                        break
                    elif choice == '' or choice == '1':
                        selected_idx = 0
                        break
                    elif choice.isdigit() and 1 <= int(choice) <= n_total:
                        selected_idx = int(choice) - 1
                        break
                    else:
                        print(f"        ✗ Invalid, enter 0-{n_total}")
                
                if selected_idx is None:
                    continue
                
                final_idx = all_indices[selected_idx]
                final_mask = masks[final_idx].squeeze().cpu().numpy()
                
                print(f"        ✓ Selected: #{selected_idx+1}")
                
                part_masks[view_name] = final_mask
                
                # Save results
                vis = image.copy()
                vis[final_mask > 0] = vis[final_mask > 0] * 0.5 + np.array(COLORS[i % len(COLORS)]) * 255 * 0.5
                
                cv2.imwrite(
                    os.path.join(self.output_dir, f"segment_{color_name}_{view_name}.png"),
                    cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
                )
                
                np.save(os.path.join(self.output_dir, f"mask_{color_name}_{view_name}.npy"), final_mask)
                print(f"        ✓ Saved")
            
            all_parts_results[color_name] = {
                'text_prompt': text_prompt,
                'point_index': i,
                'masks_2d': part_masks
            }
            
            print(f"\n  ✓ [{color_name}] Selected {len(part_masks)} views")
        
        self.segmentation_results = all_parts_results
        return all_parts_results
    
    def _show_candidates(
        self,
        image: np.ndarray,
        view_name: str,
        outputs: List,
        top_indices: np.ndarray,
        scores: np.ndarray,
        color_name: str,
        text_prompt: str
    ) -> str:
        """Show all candidates without scores for cleaner display"""
        
        # Show all candidates, not just top-5
        n_candidates = len(top_indices)
        
        # Arrange in grid (max 5 per row)
        n_cols = min(5, n_candidates)
        n_rows = (n_candidates + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        
        # Flatten axes for easier iteration
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'{view_name.upper()} - {color_name} "{text_prompt}"', fontsize=16, fontweight='bold')
        
        for i in range(len(axes)):
            if i < n_candidates:
                idx = top_indices[i]
                mask = outputs[idx]['segmentation']
                
                vis = image.copy()
                vis[mask > 0] = vis[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                
                axes[i].imshow(vis.astype(np.uint8))
                axes[i].set_title(f'#{i+1}', fontsize=16, fontweight='bold')
                axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.debug_dir, f"candidates_{color_name}_{view_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def export_parts_as_meshes(self) -> Dict[str, str]:
        """Split mesh directly using face centroid projection (from simple_segmenter)"""
        print("\n" + "="*70)
        print("Splitting Mesh by Face Centroid Projection")
        print("="*70)
        
        # Get the selected view and masks
        view_name = list(self.rendered_views.keys())[0]
        
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        n_faces = len(faces)
        
        # Get camera parameters
        cam_params = self.camera_params[view_name]
        intr = np.array(cam_params['intrinsics'])
        c2w = np.array(cam_params['c2w'])
        
        # Get masks for each part
        part_masks = {}
        for color_name, result in self.segmentation_results.items():
            if view_name in result['masks_2d']:
                part_masks[color_name] = result['masks_2d'][view_name]
        
        if not part_masks:
            raise ValueError("No masks found")
        
        first_mask = list(part_masks.values())[0]
        H, W = first_mask.shape
        
        print(f"\nUsing {view_name} view for splitting...")
        print(f"Parts: {', '.join(part_masks.keys())}")
        
        # Compute face centroids
        print("  Computing face centroids...")
        face_centroids = np.mean(vertices[faces], axis=1)
        
        # Project centroids to 2D
        print("  Projecting to 2D...")
        uv = self._project_vertices_to_2d(face_centroids, intr, c2w)
        
        # Assign faces to parts
        print("  Assigning faces...")
        part_faces = {color: [] for color in part_masks.keys()}
        
        in_bounds = 0
        out_of_bounds = 0
        
        for f_idx in range(n_faces):
            u, v = uv[f_idx]
            
            if not (0 <= u < W and 0 <= v < H):
                out_of_bounds += 1
                continue
            
            in_bounds += 1
            u_int, v_int = int(u), int(v)
            
            for color, mask in part_masks.items():
                if mask[v_int, u_int]:
                    part_faces[color].append(f_idx)
                    break
        
        print(f"\n  Projection: {in_bounds:,} in bounds, {out_of_bounds:,} out of bounds")
        for color in part_masks.keys():
            print(f"  {color}: {len(part_faces[color]):,} faces")
        
        # Extract meshes
        part_meshes = {}
        for color in part_masks.keys():
            if len(part_faces[color]) == 0:
                print(f"\n  ⚠ {color} has 0 faces!")
                continue
            
            mesh = self._extract_mesh_by_faces(part_faces[color])
            mesh_path = os.path.join(self.output_dir, f"part_{color}.glb")
            mesh.export(mesh_path)
            part_meshes[color] = mesh_path
            
            print(f"  ✓ {color}: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
        
        return part_meshes
    
    def _project_vertices_to_2d(self, vertices: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
        """Project 3D vertices to 2D (from simple_segmenter with fixed OpenGL alignment)"""
        cam_pos = c2w[:3, 3]
        R_c2w = c2w[:3, :3]
        R_w2c = R_c2w.T
        
        vertices_world = vertices - cam_pos
        vertices_cam = (R_w2c @ vertices_world.T).T
        
        # OpenGL: camera looks down -Z
        valid_depth = vertices_cam[:, 2] < 0
        
        vertices_cam_proj = vertices_cam.copy()
        vertices_cam_proj[:, 2] = -vertices_cam_proj[:, 2]
        
        uv_homog = vertices_cam_proj @ K.T
        uv = uv_homog[:, :2] / (uv_homog[:, 2:3] + 1e-8)
        
        # Flip Y: OpenGL bottom-left → image top-left
        image_height = K[1, 2] * 2
        uv[:, 1] = image_height - uv[:, 1]
        
        uv[~valid_depth] = -999999
        
        return uv
    
    def _extract_mesh_by_faces(self, face_indices: List[int]) -> trimesh.Trimesh:
        """Extract submesh by face indices (from simple_segmenter)"""
        if len(face_indices) == 0:
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
        
        selected_faces = self.mesh.faces[face_indices]
        used_vertices = np.unique(selected_faces.flatten())
        
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        new_vertices = self.mesh.vertices[used_vertices]
        new_faces = np.array([[old_to_new[v] for v in face] for face in selected_faces])
        
        return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    
    def run_interactive(self) -> Dict[str, trimesh.Trimesh]:
        """
        Run complete interactive pipeline
        
        Returns:
            Dictionary with 'lid' and 'body' meshes
        """
        print("\n" + "="*70)
        print("Complete SAM+CLIP Pipeline")
        print("="*70)
        print("\nStages:")
        print("  1. Interactive point selection")
        print("  2. Render 6 views with markers")
        print("  3. Select best view (GPT or manual)")
        print("  4. Input CLIP text prompts (GPT or manual)")
        print("  5. SAM+CLIP segmentation (GPT or manual)")
        print("  6. Project to 3D and export")
        print("="*70)
        
        # Check for cached results
        points_cache = os.path.join(self.output_dir, "selected_points.json")
        clip_cache = os.path.join(self.output_dir, "clip_prompts.json")
        
        if os.path.exists(points_cache) and os.path.exists(clip_cache):
            print("\n" + "!"*70)
            print("Found cached results!")
            print("!"*70)
            print("\nOptions:")
            print("  1. Resume from cache (skip point selection & naming) ⭐")
            print("  2. Start fresh (delete cache)")
            
            choice = input("\nSelect (1/2, default=1): ").strip()
            
            if choice != '2':
                # Load cache
                with open(points_cache, 'r') as f:
                    points_data = json.load(f)
                with open(clip_cache, 'r') as f:
                    clip_prompts = json.load(f)
                
                self.selected_points = [np.array(p['position']) for p in points_data['points']]
                self.selected_indices = [p['vertex_index'] for p in points_data['points']]
                
                print(f"\n✓ Loaded {len(self.selected_points)} points from cache")
                print(f"✓ Loaded CLIP prompts:")
                for color, prompt in clip_prompts.items():
                    print(f"  • {color}: \"{prompt}\"")
                
                skip_to_stage = 2
            else:
                print("\nDeleting cache...")
                import glob
                for pattern in ["selected_points.json", "clip_prompts.json", "view_*.png", "mask_*.npy", "segment_*.png"]:
                    for f in glob.glob(os.path.join(self.output_dir, pattern)):
                        os.remove(f)
                        print(f"  Deleted: {os.path.basename(f)}")
                skip_to_stage = 1
        else:
            skip_to_stage = 1
        
        # Stage 1: Point selection
        if skip_to_stage <= 1:
            print("\n[Stage 1/6] Interactive Point Selection")
            print("-" * 70)
            success = self.start_interactive_selection(max_points=5)
            if not success:
                raise ValueError("No points selected")
            
            self.visualize_selected_points()
        
        # Stage 2: Render views
        print("\n[Stage 2/6] Rendering 6 Views with Colored Markers")
        print("-" * 70)
        self.render_views_with_colored_markers()
        
        # Stage 3: Select best view
        print("\n[Stage 3/6] Select Best View for Segmentation")
        print("-" * 70)
        print("\nChoose selection method:")
        print("  1. GPT automatic view selection (AI chooses best view) ⭐")
        print("  2. Manual view selection (you choose view)")
        
        choice = input("\nSelect (1/2, default=1): ").strip()
        
        if choice == '2':
            # Manual view selection
            print("\nAvailable views:")
            view_list = list(self.rendered_views.keys())
            for i, view in enumerate(view_list, 1):
                print(f"  {i}. {view.upper()}")
            
            while True:
                view_choice = input(f"\nSelect view (1-{len(view_list)}): ").strip()
                if view_choice.isdigit() and 1 <= int(view_choice) <= len(view_list):
                    selected_view = view_list[int(view_choice) - 1]
                    print(f"✓ Selected: {selected_view.upper()}")
                    break
                else:
                    print(f"✗ Invalid, enter 1-{len(view_list)}")
        else:
            # GPT view selection
            print("\nUsing GPT to select best view...")
            selected_view = self._gpt_select_best_view()
            if not selected_view:
                print("  ✗ GPT failed, using default: top")
                selected_view = 'top'
            else:
                print(f"✓ GPT selected: {selected_view.upper()}")
        
        # Keep only selected view
        self.selected_view_name = selected_view
        self.rendered_views = {selected_view: self.rendered_views[selected_view]}
        self.rendered_views_marked = {selected_view: self.rendered_views_marked[selected_view]}
        
        # Stage 4: CLIP prompts
        if skip_to_stage <= 1:
            print("\n[Stage 4/6] Input CLIP Text Prompts")
            print("-" * 70)
            print("\nChoose input method:")
            print("  1. GPT auto-generation (AI generates part names) ⭐")
            print("  2. Manual input (you enter part names)")
            
            choice = input("\nSelect (1/2, default=1): ").strip()
            
            if choice == '2':
                print("\nUsing manual input...")
                clip_prompts = self.manual_input_clip_prompts()
            else:
                print("\nUsing GPT Vision API...")
                clip_prompts = self.query_gpt_for_clip_prompts()
                if not clip_prompts:
                    print("\n  ✗ GPT generation failed, switching to manual input")
                    clip_prompts = self.manual_input_clip_prompts()
            
            if not clip_prompts:
                raise ValueError("No CLIP prompts provided")
            
            self.save_intermediate_results(clip_prompts)
        else:
            print("\n[Stage 4/6] Using cached CLIP prompts")
            # clip_prompts already loaded from cache
        
        # Check if segmentation already done
        mask_cache = os.path.join(self.output_dir, f"mask_RED_{self.selected_view_name}.npy")
        if os.path.exists(mask_cache):
            print("\n" + "!"*70)
            print("Found cached segmentation masks!")
            print("!"*70)
            print("\nOptions:")
            print("  1. Use cached masks (skip segmentation) ⭐")
            print("  2. Re-segment (delete cache)")
            
            choice = input("\nSelect (1/2, default=1): ").strip()
            
            if choice != '2':
                print("\n✓ Using cached masks, skipping to export...")
                # Load masks will happen in export stage
                skip_segmentation = True
            else:
                print("\nDeleting mask cache...")
                import glob
                for f in glob.glob(os.path.join(self.output_dir, "mask_*.npy")):
                    os.remove(f)
                    print(f"  Deleted: {os.path.basename(f)}")
                skip_segmentation = False
        else:
            skip_segmentation = False
        
        if not skip_segmentation:
            # Stage 5: Select which part to segment
            print("\n[Stage 5/6] Select Part to Segment")
            print("-" * 70)
            print("\nYou only need to segment ONE part.")
            print("The other part will automatically be everything else (inverted mask).")
            print("\nAvailable parts:")
            
            part_list = []
            for i in range(len(self.selected_points)):
                color = COLOR_NAMES[i]
                name = clip_prompts.get(color, color)
                part_list.append((color, name))
                print(f"  {i+1}. {color} - \"{name}\"")
            
            print("\nChoose selection method:")
            print("  1. GPT automatic selection (AI chooses which part to segment) ⭐")
            print("  2. Manual selection (you choose which part)")
            
            choice = input("\nSelect (1/2, default=1): ").strip()
            
            if choice == '2':
                # Manual part selection
                while True:
                    part_choice = input(f"\nSelect part to segment (1-{len(part_list)}): ").strip()
                    if part_choice.isdigit() and 1 <= int(part_choice) <= len(part_list):
                        selected_color, selected_name = part_list[int(part_choice) - 1]
                        print(f"✓ Will segment: {selected_color} - \"{selected_name}\"")
                        break
                    else:
                        print(f"✗ Invalid, enter 1-{len(part_list)}")
            else:
                # GPT selects which part (usually the smaller/movable one)
                print("\nGPT selecting which part to segment...")
                selected_color = self._gpt_select_part_to_segment(clip_prompts)
                if not selected_color:
                    print("  ✗ GPT failed, using first part")
                    selected_color, selected_name = part_list[0]
                else:
                    selected_name = clip_prompts[selected_color]
                    print(f"✓ GPT chose: {selected_color} - \"{selected_name}\"")
            
            print(f"\nOther parts will use inverted mask")
            
            # Stage 5b: Segmentation
            print("\n[Stage 5b/6] SAM+CLIP Segmentation")
            print("-" * 70)
            print("\nChoose candidate selection method:")
            print("  1. GPT automatic selection (AI chooses best candidate) ⭐")
            print("  2. Manual selection (you choose best candidate)")
            
            choice = input("\nSelect (1/2, default=1): ").strip()
            
            if choice == '2':
                print("\nUsing manual candidate selection...")
                self.segment_single_part_manual(selected_color, clip_prompts)
            else:
                print("\nUsing GPT Vision for automatic candidate selection...")
                self.segment_single_part_gpt(selected_color, clip_prompts)
        else:
            # Load cached masks
            print("\nLoading cached masks...")
            self.segmentation_results = {}
            for color in COLOR_NAMES[:len(self.selected_points)]:
                mask_file = os.path.join(self.output_dir, f"mask_{color}_{self.selected_view_name}.npy")
                if os.path.exists(mask_file):
                    mask = np.load(mask_file)
                    self.segmentation_results[color] = {
                        'text_prompt': clip_prompts.get(color, color),
                        'masks_2d': {self.selected_view_name: mask}
                    }
                    print(f"  ✓ {color}: loaded cached mask")
        
        # Stage 6: Export
        print("\n[Stage 6/6] Exporting Results")
        print("-" * 70)
        part_meshes = self.export_parts_as_meshes()
        
        # Map to lid/body
        result_meshes = {}
        if "RED" in part_meshes:
            result_meshes['lid'] = load_mesh(part_meshes["RED"])
        if "GREEN" in part_meshes:
            result_meshes['body'] = load_mesh(part_meshes["GREEN"])
        
        print("\n" + "="*70)
        print("Pipeline Complete!")
        print("="*70)
        print(f"Output directory: {self.output_dir}")
        print("="*70 + "\n")
        
        return result_meshes
    
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
