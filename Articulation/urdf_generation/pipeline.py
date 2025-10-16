#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Complete URDF generation pipeline with GPT parameter optimization"""

import os
import json
import numpy as np
from pathlib import Path
import base64

from autoseg.utils.mesh_utils import load_mesh, save_mesh, translate_mesh
from autoseg.urdf_generation.hinge_detector import HingeDetector
from autoseg.urdf_generation.urdf_builder import URDFBuilder

try:
    import openai
    HAS_OPENAI = True
except:
    HAS_OPENAI = False


class URDFGenerationPipeline:
    """Complete pipeline for generating URDF from segmented parts"""
    
    def __init__(self, openbox_path: str, lid_path: str, body_path: str, output_dir: str):
        """
        Initialize pipeline
        
        Args:
            openbox_path: Path to complete object (for reference)
            lid_path: Path to lid mesh
            body_path: Path to body mesh
            output_dir: Output directory for URDF and meshes
        """
        self.openbox_path = openbox_path
        self.lid_path = lid_path
        self.body_path = body_path
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Loading meshes...")
        self.openbox = load_mesh(openbox_path)
        self.lid = load_mesh(lid_path)
        self.body = load_mesh(body_path)
        
        print(f"  Openbox: {len(self.openbox.vertices):,} vertices")
        print(f"  Lid: {len(self.lid.vertices):,} vertices")
        print(f"  Body: {len(self.body.vertices):,} vertices")
    
    def query_gpt_for_urdf_params(self, segmentation_dir: str = None) -> dict:
        """Use GPT to analyze parts and recommend URDF parameters"""
        
        if not HAS_OPENAI:
            print("  OpenAI not available, using default parameters")
            return {}
        
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("  OPENAI_API_KEY not set, using default parameters")
            return {}
        
        openai.api_key = api_key
        
        print("\nQuerying GPT for URDF parameters...")
        
        # Load rendered images from segmentation
        print("  Loading rendered views...")
        image_contents = []
        
        if segmentation_dir:
            # Load marked views (with colored markers showing parts)
            for vname in ['top', 'bottom']:
                img_path = os.path.join(segmentation_dir, f"view_{vname}_marked.png")
                if os.path.exists(img_path):
                    with open(img_path, 'rb') as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                        image_contents.append({
                            "type": "text",
                            "text": f"\n--- {vname.upper()} VIEW (with colored part markers) ---"
                        })
                        image_contents.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                                "detail": "high"
                            }
                        })
        
        if image_contents:
            print(f"  ✓ Loaded {len(image_contents)//2} rendered views")
        else:
            print("  ⚠ No images found, GPT will work without visual context")
        
        # Build prompt
        prompt = f"""You are a robotics expert specializing in articulated object modeling and URDF generation.

TASK: Analyze this articulated object from the rendered views and recommend optimal URDF parameters.

The object is shown in the images above with colored markers:
- RED marker: Point on first part
- GREEN marker: Point on second part

The object has been segmented into two parts (lid and body).

Your task:
1. Analyze the images to understand the object type (box, laptop, door, etc.)
2. Determine the realistic range of motion (how far it can open)
3. Determine which part should be BASE (fixed) and which should be MOVABLE
4. Recommend physical parameters for realistic simulation

IMPORTANT: Look carefully at the images to determine:
- How the parts connect (hinged? sliding?)
- Maximum realistic opening angle
- Size and weight proportions

Provide your recommendations in JSON format:
{{
  "object_type": "box|laptop|door|scissors|etc",
  "base_link": "lid|body",
  "movable_link": "lid|body",
  "joint_limits": {{
    "lower": -0.785,
    "upper": 0.785,
    "comment": "In radians. Common ranges: -45° to +45° (±0.785), 0° to 90° (0-1.57), 0° to 120° (0-2.09)"
  }},
  "dynamics": {{
    "effort": 5.0,
    "velocity": 2.0,
    "friction": 0.5,
    "damping": 0.2,
    "comment": "Adjust based on object size and weight"
  }},
  "mass": {{
    "base": 0.5,
    "movable": 0.2,
    "comment": "Estimate from visual size"
  }},
  "reasoning": "Explain why you chose these limits and parameters based on what you see in the images"
}}

Guidelines:
- BASE: Usually larger, heavier, more stable part (e.g., box body, laptop base)
- MOVABLE: Usually smaller, lighter part that rotates (e.g., box lid, laptop screen)
- Friction: 0.1-2.0 (higher = more resistance to motion)
- Damping: 0.1-1.0 (higher = slower, more controlled motion)
- Joint limits: IMPORTANT - Analyze the images to determine realistic opening range
  - Box lid: typically 90-120 degrees (1.57-2.09 rad)
  - Laptop screen: typically 90-135 degrees (1.57-2.36 rad)
  - Door: typically 90-180 degrees (1.57-3.14 rad)

Return ONLY the JSON, no additional text.
"""

        try:
            # Build message with images
            if image_contents:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *image_contents
                    ]
                }]
            else:
                messages = [{"role": "user", "content": prompt}]
            
            print("  Querying GPT with visual context...")
            gpt_model = os.environ.get('OPENAI_GPT_MODEL', 'gpt-4-turbo')
            response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )
            
            gpt_output = response['choices'][0]['message']['content']
            
            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', gpt_output, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group())
                
                print("\n  ✓ GPT recommendations (based on visual analysis):")
                print(f"    Object type: {params.get('object_type', 'unknown')}")
                print(f"    Base link: {params.get('base_link', 'body')}")
                print(f"    Movable link: {params.get('movable_link', 'lid')}")
                
                lower = params['joint_limits']['lower']
                upper = params['joint_limits']['upper']
                print(f"    Joint limits: {lower:.2f} - {upper:.2f} rad ({np.degrees(lower):.1f}° - {np.degrees(upper):.1f}°)")
                
                print(f"    Effort: {params['dynamics']['effort']:.2f} N·m")
                print(f"    Velocity: {params['dynamics']['velocity']:.2f} rad/s")
                print(f"    Friction: {params['dynamics']['friction']:.2f}")
                print(f"    Damping: {params['dynamics']['damping']:.2f}")
                print(f"\n    Reasoning: {params.get('reasoning', 'N/A')}")
                
                return params
            else:
                print("  ⚠ Failed to parse GPT response")
                return {}
                
        except Exception as e:
            print(f"  ⚠ GPT call failed: {e}")
            return {}
    
    def generate(
        self,
        robot_name: str = "articulated_object",
        limit_lower: float = None,
        limit_upper: float = None,
        effort: float = None,
        velocity: float = None,
        friction: float = None,
        damping: float = None,
        use_gpt: bool = True
    ) -> str:
        """
        Generate URDF file
        
        Args:
            robot_name: Name for the robot
            limit_lower: Joint lower limit (radians), None=use GPT/default
            limit_upper: Joint upper limit (radians), None=use GPT/default
            effort: Maximum effort (N·m), None=use GPT/default
            velocity: Maximum velocity (rad/s), None=use GPT/default
            friction: Friction coefficient, None=use GPT/default
            damping: Damping coefficient, None=use GPT/default
            use_gpt: Whether to use GPT for parameter recommendation
        
        Returns:
            Path to generated URDF file
        """
        
        # Query GPT for parameters if enabled
        gpt_params = {}
        if use_gpt and any(p is None for p in [limit_lower, limit_upper, effort, velocity, friction, damping]):
            print("\n" + "="*70)
            print("GPT Parameter Recommendation")
            print("="*70)
            print("\nChoose parameter source:")
            print("  1. Use GPT recommendations (automatic)")
            print("  2. Use default/manual parameters")
            
            choice = input("\nSelect (1/2, default=1): ").strip()
            
            if choice != '2':
                # Try to find segmentation directory
                seg_dir = os.path.join(os.path.dirname(self.output_dir), 'segmentation')
                if not os.path.exists(seg_dir):
                    seg_dir = None
                
                gpt_params = self.query_gpt_for_urdf_params(segmentation_dir=seg_dir)
        
        # Apply GPT parameters or use defaults
        if gpt_params:
            # Determine which is base and which is movable
            base_name = gpt_params.get('base_link', 'body')
            movable_name = gpt_params.get('movable_link', 'lid')
            
            # Swap if needed
            if base_name == 'lid':
                print(f"\n  ⚠ GPT recommends lid as base, swapping parts...")
                self.lid, self.body = self.body, self.lid
            
            limit_lower = limit_lower if limit_lower is not None else gpt_params['joint_limits']['lower']
            limit_upper = limit_upper if limit_upper is not None else gpt_params['joint_limits']['upper']
            effort = effort if effort is not None else gpt_params['dynamics']['effort']
            velocity = velocity if velocity is not None else gpt_params['dynamics']['velocity']
            friction = friction if friction is not None else gpt_params['dynamics']['friction']
            damping = damping if damping is not None else gpt_params['dynamics']['damping']
        else:
            # Use defaults (-45 to +45 degrees)
            limit_lower = limit_lower if limit_lower is not None else -0.785  # -45 degrees
            limit_upper = limit_upper if limit_upper is not None else 0.785   # +45 degrees
            effort = effort if effort is not None else 5.0
            velocity = velocity if velocity is not None else 2.0
            friction = friction if friction is not None else 0.5
            damping = damping if damping is not None else 0.2
        
        print(f"\nUsing parameters:")
        print(f"  Joint limits: {limit_lower:.2f} - {limit_upper:.2f} rad ({np.degrees(limit_lower):.1f}° - {np.degrees(limit_upper):.1f}°)")
        print(f"  Effort: {effort:.2f} N·m")
        print(f"  Velocity: {velocity:.2f} rad/s")
        print(f"  Friction: {friction:.2f}")
        print(f"  Damping: {damping:.2f}")
        
        # Step 1: Detect hinge
        print("\n" + "="*70)
        print("Detecting hinge position and axis...")
        print("="*70)
        detector = HingeDetector(self.lid, self.body)
        hinge_info = detector.detect()
        
        print(f"  Position: {hinge_info['position']}")
        print(f"  Axis: {hinge_info['axis']}")
        print(f"  Axis confidence: {hinge_info['axis_confidence']:.2%}")
        print(f"  Min distance: {hinge_info['min_distance']:.4f}")
        
        # Step 2: Translate parts to origin
        print("\nTranslating parts to origin...")
        translation = detector.get_translation_to_origin()
        
        lid_centered = translate_mesh(self.lid, translation)
        body_centered = translate_mesh(self.body, translation)
        
        # Save transformed meshes
        lid_mesh_path = os.path.join(self.output_dir, "lid_centered.glb")
        body_mesh_path = os.path.join(self.output_dir, "body_centered.glb")
        
        save_mesh(lid_centered, lid_mesh_path)
        save_mesh(body_centered, body_mesh_path)
        
        print(f"  Saved: {os.path.basename(lid_mesh_path)}")
        print(f"  Saved: {os.path.basename(body_mesh_path)}")
        
        # Step 3: Build URDF
        print("\nBuilding URDF...")
        builder = URDFBuilder(robot_name)
        
        # Add base link (body)
        builder.add_link(
            name="body",
            mesh_file="body_centered.glb",
            color=[0.8, 0.6, 0.4, 1.0],
            mass=0.5
        )
        
        # Add movable link (lid)
        builder.add_link(
            name="lid",
            mesh_file="lid_centered.glb",
            color=[0.6, 0.8, 0.4, 1.0],
            mass=0.2
        )
        
        # Add revolute joint
        builder.add_revolute_joint(
            name="hinge",
            parent="body",
            child="lid",
            origin_xyz=[0, 0, 0],
            origin_rpy=[0, 0, 0],
            axis=hinge_info['axis'],
            limit_lower=limit_lower,
            limit_upper=limit_upper,
            effort=effort,
            velocity=velocity,
            friction=friction,
            damping=damping
        )
        
        # Generate URDF file
        urdf_path = os.path.join(self.output_dir, f"{robot_name}.urdf")
        builder.save(urdf_path)
        print(f"  URDF: {os.path.basename(urdf_path)}")
        
        # Step 4: Save metadata
        metadata = {
            "robot_name": robot_name,
            "files": {
                "urdf": f"{robot_name}.urdf",
                "body_mesh": "body_centered.glb",
                "lid_mesh": "lid_centered.glb"
            },
            "hinge": {
                "original_position": hinge_info['position'].tolist(),
                "axis": hinge_info['axis'].tolist(),
                "axis_confidence": float(hinge_info['axis_confidence']),
                "translation_applied": translation.tolist()
            },
            "joint_limits": {
                "lower": float(limit_lower),
                "upper": float(limit_upper),
                "lower_deg": float(np.degrees(limit_lower)),
                "upper_deg": float(np.degrees(limit_upper))
            },
            "dynamics": {
                "effort": float(effort),
                "velocity": float(velocity),
                "friction": float(friction),
                "damping": float(damping)
            }
        }
        
        # Add GPT recommendations if available
        if gpt_params:
            metadata["gpt_recommendations"] = {
                "object_type": gpt_params.get('object_type', 'unknown'),
                "reasoning": gpt_params.get('reasoning', ''),
                "parameters_used": True
            }
        else:
            metadata["gpt_recommendations"] = {
                "parameters_used": False
            }
        
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata: {os.path.basename(metadata_path)}")
        
        return urdf_path

