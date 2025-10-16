## MLLM-driven Articulation Inference

This tool provides a complete pipeline to convert a static 3D model (.glb) of an articulated object into a fully functional URDF file, ready for physics simulation. The process combines interactive 3D segmentation, automatic hinge detection, and intelligent physical parameter estimation.

The core workflow is:
GLB Model → Interactive Segmentation → Part Extraction → Hinge Detection → URDF Generation

1. **Setup**

**Download SAM Checkpoint**.
Before running the script, you must first download the [Segment Anything Model (SAM) checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth). The default model is ViT-H.


**Configure OpenAI API Key**.
To enable the MLLM features for physics inference, you need to configure your OpenAI API key.

```bash
export OPENAI_API_KEY=""
```

Using GPT's "Thinking Mode" is crucial for generating high-quality, practical caption for segmentation. 

2. **Example**

The main process involves segmenting the object and then generating the URDF. Let's segment the provided openbox.glb model. Run the following command in your terminal. This command launches the interactive segmentation tool and specifies your API key for subsequent steps.

```bash
cd Articulation
python articulation_inference.py model.glb --interactive
```

## MLLM-driven Physics Estimation

This tool uses a Multi-modal Large Language Model (MLLM) to automatically estimate an object's key physical attributes, such as Density, Young's modulus, and Poisson's ratio. These estimations are then uesd in our simulation process.

1. **Example**

```bash
cd Articulation
python physics_estimation.py --input openbox_output/segmentation/view_top.png
```