import os
import sys
import argparse
from pathlib import Path

autoseg_root = Path(__file__).parent
if str(autoseg_root.parent) not in sys.path:
    sys.path.insert(0, str(autoseg_root.parent))

# Now import from autoseg package
from segmentation.interactive_segmenter import InteractiveSegmenter
from urdf_generation.pipeline import URDFGenerationPipeline
from utils.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Articulated Object Segmentation and URDF Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: Interactive segmentation with default parameters
  python main.py model.glb --interactive
  
  # With custom URDF parameters
  python main.py model.glb --interactive --limit-upper 1.57 --friction 0.8
  
  # Use specific GPT model
  python main.py model.glb --interactive --api-key sk-... --gpt-model gpt-4o
  
  # Disable all GPT features (manual only)
  python main.py model.glb --interactive --no-gpt
  
  # Using pre-segmented parts
  python main.py model.glb --lid lid.glb --body body.glb
  
  # Custom output and robot name
  python main.py model.glb --interactive --output ./my_output --robot-name my_box
  
  # Full custom parameters
  python main.py model.glb --interactive --limit-lower 0 --limit-upper 1.57 \\
    --effort 10.0 --velocity 3.0 --friction 0.8 --damping 0.3
  
  # Use different SAM model
  python main.py model.glb --interactive --sam-model vit_l \\
    --sam-checkpoint sam_vit_l.pth
        """
    )
    
    # Input files
    parser.add_argument("input", help="Input GLB file (complete articulated object)")
    parser.add_argument("--lid", help="Pre-segmented lid GLB file (optional)")
    parser.add_argument("--body", help="Pre-segmented body GLB file (optional)")
    
    # Operation mode
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Enable interactive point selection for segmentation")
    parser.add_argument("--skip-segmentation", action="store_true",
                       help="Skip segmentation (requires --lid and --body)")
    parser.add_argument("--use-sam-clip", action="store_true",
                       help="Use full SAM+CLIP pipeline (slower but more accurate)")
    
    # SAM configuration
    parser.add_argument("--sam-checkpoint", default="sam.pth",
                       help="SAM model checkpoint path")
    
    # URDF parameters
    parser.add_argument("--robot-name", default="articulated_object",
                       help="Name for the generated robot/URDF")
    parser.add_argument("--limit-lower", type=float, default=None,
                       help="Joint lower limit (radians, default=GPT or -0.785≈-45°)")
    parser.add_argument("--limit-upper", type=float, default=None,
                       help="Joint upper limit (radians, default=GPT or 0.785≈+45°)")
    parser.add_argument("--effort", type=float, default=None,
                       help="Maximum effort/torque (N·m, default=GPT or 5.0)")
    parser.add_argument("--velocity", type=float, default=None,
                       help="Maximum velocity (rad/s, default=GPT or 2.0)")
    parser.add_argument("--friction", type=float, default=None,
                       help="Friction coefficient (default=GPT or 0.5)")
    parser.add_argument("--damping", type=float, default=None,
                       help="Damping coefficient (default=GPT or 0.2)")
    parser.add_argument("--use-gpt-params", action="store_true",
                       help="Use GPT to recommend URDF parameters automatically")
    parser.add_argument("--no-gpt", action="store_true",
                       help="Disable all GPT features")
    parser.add_argument("--api-key", default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env variable)")
    parser.add_argument("--gpt-model", default="gpt-4-turbo",
                       help="GPT model to use")
    
    # Output
    parser.add_argument("--output", "-o", default=None,
                       help="Output directory (default: same as input file)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set OpenAI API key if provided
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
        print(f"Using provided API key")
    elif os.environ.get('OPENAI_API_KEY'):
        print(f"Using API key from environment")
    else:
        if not args.no_gpt:
            print("Note: OPENAI_API_KEY not set. GPT features will prompt for manual fallback.")
            print("      Use --api-key <key> or export OPENAI_API_KEY=<key>")
            print("      Or use --no-gpt to disable GPT features\n")
    
    # Set GPT model
    if args.gpt_model:
        os.environ['OPENAI_GPT_MODEL'] = args.gpt_model
        print(f"Using GPT model: {args.gpt_model}")
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if args.skip_segmentation and (not args.lid or not args.body):
        print("Error: --skip-segmentation requires both --lid and --body")
        return 1
    
    # Create config
    config = Config(
        input_file=args.input,
        output_dir=args.output,
        sam_checkpoint=args.sam_checkpoint,
        robot_name=args.robot_name,
        verbose=args.verbose
    )
    
    print("\n" + "="*70)
    print("Articulated Object Segmentation & URDF Generation Pipeline")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Output: {config.output_dir}")
    print(f"Mode: {'Interactive' if args.interactive else 'Automatic'}")
    print("="*70 + "\n")
    
    try:
        # Stage 1: Segmentation
        if not args.skip_segmentation:
            print("\n[Stage 1/2] Segmentation")
            print("-" * 70)
            
            segmenter = InteractiveSegmenter(
                glb_path=args.input,
                sam_checkpoint=config.sam_checkpoint_path,
                output_dir=config.segmentation_dir
            )
            
            # Run segmentation (always interactive now)
            segmented_parts = segmenter.run_interactive()
            
            if not segmented_parts or len(segmented_parts) < 2:
                print("Error: Segmentation failed or insufficient parts found")
                return 1
            
            # Export parts with their user-defined names
            exported_files = segmenter.export_parts(segmented_parts, config.parts_dir)
            
            # Try to identify lid and body from part names
            part_names = list(segmented_parts.keys())
            
            # Simple heuristic: look for common keywords
            lid_candidates = ['lid', 'cover', 'top', 'door', 'flap']
            body_candidates = ['body', 'base', 'bottom', 'frame', 'container']
            
            lid_name = None
            body_name = None
            
            for name in part_names:
                if any(keyword in name.lower() for keyword in lid_candidates):
                    lid_name = name
                elif any(keyword in name.lower() for keyword in body_candidates):
                    body_name = name
            
            # Fallback: use first two parts
            if not lid_name:
                lid_name = part_names[0]
            if not body_name:
                body_name = part_names[1] if len(part_names) > 1 else part_names[0]
            
            lid_path = exported_files[lid_name]
            body_path = exported_files[body_name]
            
            print(f"\nIdentified parts for URDF:")
            print(f"  Lid (movable): {lid_name} → {lid_path}")
            print(f"  Body (fixed): {body_name} → {body_path}")
            
        else:
            lid_path = args.lid
            body_path = args.body
            print(f"\n[Stage 1/2] Using pre-segmented parts")
            print(f"  Lid: {lid_path}")
            print(f"  Body: {body_path}")
        
        # Stage 2: URDF Generation
        print("\n[Stage 2/2] URDF Generation")
        print("-" * 70)
        
        pipeline = URDFGenerationPipeline(
            openbox_path=args.input,
            lid_path=lid_path,
            body_path=body_path,
            output_dir=config.urdf_dir
        )
        
        urdf_path = pipeline.generate(
            robot_name=args.robot_name,
            limit_lower=args.limit_lower,
            limit_upper=args.limit_upper,
            effort=args.effort,
            velocity=args.velocity,
            friction=args.friction,
            damping=args.damping,
            use_gpt=not args.no_gpt  # Enable GPT unless --no-gpt is set
        )
        
        # Summary
        print("\n" + "="*70)
        print("Pipeline Complete!")
        print("="*70)
        print(f"\nGenerated Files:")
        print(f"  URDF: {urdf_path}")
        print(f"  Output Directory: {config.output_dir}")
        print(f"\nNext Steps:")
        print(f"  1. Test in PyBullet: python -m utils.test_pybullet {urdf_path}")
        print(f"  2. Visualize: python -m utils.visualize {config.urdf_dir}")
        print(f"  3. See README.md for ROS/Gazebo usage")
        print("="*70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

