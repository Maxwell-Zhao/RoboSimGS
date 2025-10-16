import openai
import base64
import json
import io



# PRED_CAND_MATS_DENSITY_SYS_MSG_4V = """You will be provided with an image of an object and its caption. The caption will be delimited with quotes ("). Based on the image and caption, help me to guess the object's MPM physical parameters. You must provide your answer in the following JSON format, as it will be parsed by a code script later. 
# Format Requirement:
# You must provide your answer in the following JSON format, as it will be parsed by a code script later. Your answer must look like:
# {
#     "Density (kg/m³)": value, 
#     "Young's modulus (MPa)": value,
#     "Poisson's ratio": value,
# } 
# Do not include any other text in your answer.
# """

# PRED_CAND_MATS_DENSITY_SYS_MSG_4V = """You will be provided with an image of an object and its caption. The caption will be delimited with quotes ("). Based on the image and caption, help me to speculate the object's physical parameters that matchs the label class and material of the object. The physical parameters will be used in a simulation called Material Point Method. You must provide your answer in the following JSON format, as it will be parsed by a code script later. 
# Format Requirement:
# You must provide your answer in the following JSON format, as it will be parsed by a code script later. Your answer must look like:
# {
#     "object's label": value,
#     "object's Young's modulus (MPa) ": value,
# } 
# Do not include any other text in your answer.
# noted:
# Young's modulus (MPa) is a measure of the stiffness of a material. It is used to describe the elastic properties of a material. The higher the Young's modulus, the stiffer the material. 
# """
# GPT
# PRED_CAND_MATS_DENSITY_SYS_MSG_4V = """You will be provided with an image of an object. Help me to speculate the object's MPM physical parameters. You must provide your answer in the following JSON format, as it will be parsed by a code script later. 
# Format Requirement:
# You must provide your answer in the following JSON format, as it will be parsed by a code script later. Your answer must look like:
# {
#     "Density (kg/m³)": value, 
#     "Young's modulus (MPa)": value,
#     "Poisson's ratio": value,
# } 
# Do not include any other text in your answer.
# """
# # GPT+BLIP
# PRED_CAND_MATS_DENSITY_SYS_MSG_4V = """You will be provided with an image of an object and its caption. The caption will be delimited with quotes ("). Based on the image and caption, help me to speculate the object's MPM physical parameters based on the note You must provide your answer in the following JSON format, as it will be parsed by a code script later. 
# Format Requirement:
# You must provide your answer in the following JSON format, as it will be parsed by a code script later. Your answer must look like:
# {
#     "Density (kg/m³)": value, 
#     "Young's modulus (MPa)": value 
#     "Poisson's ratio": value,
# } 
# noted:
# Young's modulus (MPa) is a measure of the stiffness of a material. e.g. a ball's young's modulus is 5e4 to 5e6 MPa.
# Do not include any other text in your answer.

# """

#full
PRED_CAND_MATS_DENSITY_SYS_MSG_4V = """You will be provided with an image of an object and its caption. The caption will be delimited with quotes ("). Based on the image and caption, help me to speculate the object's material category.
You must provide your answer in the following JSON format, as it will be parsed by a code script later. Your answer must look like:
{
    1. Wheter the object is a rigid object or a non-rigid or a elastic object.
    2. predict three most suitable material for the object considering their physical parameters .
    3. predict the object's density value range(kg/m^3).
}
Do not include any other text in your answer. 
The material name range are listed below:
gelatin, rubber, leather, nylon, elastic, wood, plant fiber, metal
"""

FINE_SYS_MSG_4V = """You will be provided with an image of an object and a caption. Based on the image and message, help me to predict the object's physical parameters. You must provide your answer in the following JSON format, as it will be parsed by a code script later. 
Format Requirement:
You must provide your answer in the following JSON format, as it will be parsed by a code script later. Your answer must look like:
{
    "Young's modulus (MPa)[5e4, 5e6]": value,
    "Poisson's ratio": value,
    "Density (kg/m³)[1e2, 1e3]": value,
} 
Do not include any other text in your answer.
noted:
Young's modulus (MPa), mainly between 5e4 to 5e6, is a measure of the stiffness of a material. It is used to describe the elastic properties of a material. The higher the Young's modulus, the stiffer the material. 

"""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_tensor(tensor_image):
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu()

    if len(tensor_image.shape) == 3:
        tensor_image = tensor_image.unsqueeze(0)

    # 把tensor转换为NumPy数组，并去掉批处理维度 (假设是 (1, H, W, C) 或 (C, H, W))
    np_image = tensor_image.squeeze(0).permute(1, 2, 0).numpy()

    # 将NumPy数组转换为PIL图像
    image = Image.fromarray((np_image * 255).astype(np.uint8))

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")  # 保存为PNG格式

    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def gpt4v_candidate_materials(image,caption, seed=100, msg=None):
    if msg is not None:
        sys_msg = msg
    else:
        sys_msg = PRED_CAND_MATS_DENSITY_SYS_MSG_4V

    base64_image = encode_image_from_tensor(image)

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content":  '"%s"' % caption},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},]}
        ],
        request_timeout=30,
        max_tokens=300,
        seed=seed,
        # response_format={"type": "json_object"},
    )
    return response['choices'][0]['message']['content']

def gpt4v_fine_materials_params(image,caption, seed=100, msg=None):
    if msg is not None:
        sys_msg = msg
    else:
        sys_msg = FINE_SYS_MSG_4V

    base64_image = encode_image_from_tensor(image)

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content":  '"%s"' % caption},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},]}
        ],
        request_timeout=30,
        max_tokens=300,
        seed=seed,
        # response_format={"type": "json_object"},
    )
    return response['choices'][0]['message']['content']
def parse_material_json(matjson, max_n=5, field_name='mass density (kg/m^3)'):
    desc_and_mats = json.loads(matjson)
    if 'description' not in desc_and_mats or 'materials' not in desc_and_mats:
        print('bad format %s' % matjson)
        return None
    mat_names = []
    mat_vals = []
    for mat in desc_and_mats['materials']:
        if 'name' not in mat or field_name not in mat:
            print('bad format %s' % matjson)
            return None
        mat_name = mat['name']
        mat_names.append(mat_name.lower())  # force lowercase
        values = mat[field_name]
        # Value may or may not be a range
        splitted = values.split('-')
        try:
            float(splitted[0])
        except ValueError:
            print('value cannot be converted to float %s' % matjson)
            return None
        if len(splitted) == 2:
            mat_vals.append([float(splitted[0]), float(splitted[1])])
        elif len(splitted) == 1:
            mat_vals.append([float(splitted[0]), float(splitted[0])])
        else:
            print('bad format %s' % matjson)
            return None
    return desc_and_mats['description'], mat_names, mat_vals

import json
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration


CAPTIONING_PROMPT = "Question: Give a detailed description of the object. Answer:"

def load_blip2(model_name, device='cuda'):
    processor = AutoProcessor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
    return model, processor

def generate_text(img, model, processor, prompt=CAPTIONING_PROMPT, device='cuda'):
    if prompt is not None:
        inputs = processor(img, text=prompt, return_tensors="pt").to(device, torch.float16)
    else:
        inputs = processor(img, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=30)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

def predict_caption(imgs, vqa_model, vqa_processor):
    captions = []
    for i, img in enumerate(imgs):
        with torch.no_grad():
            caption = generate_text(img, vqa_model, vqa_processor, device='cuda')
        info = {'idx': str(i), 'caption': caption}
        captions.append(info)

    json_dir = 'caption.json'
    with open(json_dir, 'w') as f:
        json.dump(captions, f, indent=4)

    return captions


import os
import time
import json
import openai
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.nerf2physic_utils import load_images, get_scenes_list

BASE_SEED = 100

def gpt_wrapper(gpt_fn, max_tries=10, sleep_time=3):
    """Wrap gpt_fn with error handling and retrying."""
    tries = 0
    # sleep to avoid overloading openai api
    time.sleep(sleep_time)
    try:
        gpt_response = gpt_fn(BASE_SEED + tries)
        result = True
    except Exception as error:
        print('error:', error)
        result = None
    while result is None and tries < max_tries:
        tries += 1
        time.sleep(sleep_time)
        print('retrying...')
        try:
            gpt_response = gpt_fn(BASE_SEED + tries)
        except:
            result = None
    return gpt_response

def show_img_to_caption(scene_dir, idx_to_caption):
    img_dir = os.path.join(scene_dir, 'images')
    imgs = load_images(img_dir, bg_change=None, return_masks=False)
    img_to_caption = imgs[idx_to_caption]
    plt.imshow(img_to_caption)
    plt.show()
    plt.close()
    return

def predict_candidate_materials(caption, image, show=False, msg=None):

    gpt_fn = lambda seed: gpt4v_candidate_materials(caption, image, seed=seed, msg=msg)
    MPM_gpt = gpt_wrapper(gpt_fn)
    return MPM_gpt

def predict_fine_materials(image, caption, show=False, msg=None):

    gpt_fn = lambda seed: gpt4v_fine_materials_params(image, caption, seed=seed, msg=msg)
    MPM_gpt = gpt_wrapper(gpt_fn)
    return MPM_gpt