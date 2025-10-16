import os
import numpy as np
import math
import json
import torch
import openai
from utils.gpt_utils import load_blip2, predict_caption,predict_candidate_materials, predict_fine_materials
import argparse
import os
from dotenv import load_dotenv
import cv2
import sys
from pathlib import Path
autoseg_root = Path(__file__).parent
if str(autoseg_root.parent) not in sys.path:
    sys.path.insert(0, str(autoseg_root.parent))
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def main():
    parser = argparse.ArgumentParser(
        description="Physics Estimation for Articulated Objects using MLLM",
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
    parser.add_argument("input", help="Input image file (rendered view of the articulated object)")
    parser.add_argument("blip", default="Salesforce/blip2-opt-2.7b", help="blip model dir")
    parser.add_argument("clip", default="openai/clip-vit-base-patch32", help="clip model dir")

    
    args = parser.parse_args()


    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ["all_proxy"]=""
    load_dotenv()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    # print(OPENAI_API_KEY)
    openai.api_key = OPENAI_API_KEY
    image = cv2.imread(args.input)

    blip2_model_dir = args.blip
    images = [image]
    model, processor = load_blip2(blip2_model_dir, device='cuda')
    caption_info = predict_caption(images, model, processor)
    print("caption_info:", caption_info)
    caption1 = caption_info[0]['caption']

    coarse_prediction = predict_candidate_materials(torch.tensor(image).permute(2, 0, 1), caption1) # until now we get the coarse_MPM
    print(coarse_prediction)
    coarse_prediction = json.loads(coarse_prediction)
    print(coarse_prediction)
    if 'rigid' in coarse_prediction['1']:
        use_mpm = True
        use_pbd = False
    else:
        use_mpm = False
        use_pbd = True

    model = CLIPModel.from_pretrained(args.clip)
    processor = CLIPProcessor.from_pretrained(args.clip)

    object_key, material_key, density_key = list(coarse_prediction.keys())[0],list(coarse_prediction.keys())[1], list(coarse_prediction.keys())[2]
    texts = coarse_prediction[material_key]
    density = coarse_prediction[density_key]
    print('text', texts)

    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
 
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1).numpy()

    max_prob = 0
    for text, prob in zip(texts, probs[0]):
        print(f"Text: '{text}' - Similarity with image: {prob:.4f}")
        if prob > max_prob:
            max_prob = prob
            max_text = text

    print("max_text:", max_text)
    
    if use_mpm:
        # "Density (kg/m³)": value, 
        # "Young's modulus (MPa)": value,
        # "Poisson's ratio": value,

        # get mpm from the material table
        table_path = 'Physical/material_table.json'
        with open(table_path, 'r') as f:
            materials_table = json.load(f)['materials']
            for material in materials_table:
                if material['name'] == max_text:
                    coarse_mpm = material
                    break
        print('coarse_mpm',coarse_mpm)
        PRED_CAND_MATS_DENSITY_SYS_MSG_4V = """You will be provided with an image of an object and its caption.The caption contains the object's name, material and coarse estimated MPM parameters.
                                            Help me to speculate the object's fine MPM physical parameters.
                                            Do not directly use the coarse MPM parameters, but use them as a reference.
                                            You must provide your answer in the following JSON format, as it will be parsed by a code script later. 
                                            Format Requirement:
                                            {
                                                "Density (kg/m³)": value, 
                                                "Young's modulus (MPa)": value,
                                                "Poisson's ratio": value,
                                            } 
                                            Do not include any other text in your answer.
                                            """
        caption1 = f"A {coarse_prediction[object_key]} carnation made of {max_text}, its density is around {density}, its coarse MPM parameters are {coarse_mpm}"
        print("caption1:", caption1)
        fine_prediction = predict_fine_materials(torch.tensor(image).permute(2, 0, 1), caption1, msg=PRED_CAND_MATS_DENSITY_SYS_MSG_4V) 
        # until now we get the coarse_MPM
        print(fine_prediction)
    if use_pbd:
        # Density (kg/m^2): 
        # static_friction: float, optional
        #     Static friction coefficient. Represents the resistance to the start of sliding motion between two contacting particles.
        #     In collision resolution, it determines how much tangential force can be applied before sliding begins. Default is 0.15.
        # kinetic_friction: float, optional
        #     Kinetic (Dynamic) Friction Coefficient. Represents the resistance during sliding motion between two contacting particles.
        #     Applied when particles are already sliding; limits the tangential force to simulate energy loss due to friction. Default is 0.0.
        # stretch_compliance: float, optional
        #     The stretch compliance (m/N). Controls the softness of the stretch constraint between particles.
        #     Low values correspond to very stiff; enforces near-constant distance. High values correspond to softer response; more stretch allowed. Default is 0.0.

        table_path = 'Physical/material_table_pbd.json'
        with open(table_path, 'r') as f:
            materials_table = json.load(f)['materials']
            for material in materials_table:
                if material['name'] == max_text:
                    coarse_pbd = material
                    break
        PRED_CAND_MATS_DENSITY_SYS_MSG_4V = """You will be provided with an image of an object and its caption.The caption contains the object's name, material and coarse estimated PBD parameters.
                                            Help me to speculate the object's fine PBD physical parameters.
                                            Do not directly use the coarse PBD parameters, but use them as a reference.
                                            You must provide your answer in the following JSON format, as it will be parsed by a code script later. 
                                            Format Requirement:
                                            {
                                                "Density (kg/m³)": value, 
                                                "static_friction (MPa)": value,
                                                "kinetic_friction": value,
                                                "stretch_compliance": value,
                                            } 
                                            Do not include any other text in your answer.
                                            """
        caption1 = f"A {coarse_prediction[object_key]} carnation made of {max_text}, its density is around {density}, its coarse PBD parameters are {coarse_pbd}"
        print("caption1:", caption1)
        fine_prediction = predict_fine_materials(torch.tensor(image).permute(2, 0, 1), caption1, msg=PRED_CAND_MATS_DENSITY_SYS_MSG_4V) 
        # until now we get the coarse_MPM
        print(fine_prediction)


if __name__ == "__main__":
    sys.exit(main())