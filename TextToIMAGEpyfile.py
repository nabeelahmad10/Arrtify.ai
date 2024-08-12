# model.py

from googletrans import Translator
import torch
from diffusers import StableDiffusionPipeline
import clip
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (900, 900)
    image_gen_guidance_scale = 9

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='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', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)

clip_model, preprocess = clip.load("ViT-B/32", device=CFG.device)

def generate_image(prompt):
    image = image_gen_model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image

def get_translation(text, dest_lang):
    translator = Translator()
    translated_text = translator.translate(text, dest=dest_lang)
    return translated_text.text

def calculate_clip_score(prompt, image):
    image = preprocess(image).unsqueeze(0).to(CFG.device)
    text = clip.tokenize([prompt]).to(CFG.device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).item()
    return similarity

def calculate_ssim(image1, image2):
    image1 = np.array(image1.convert('L'))
    image2 = np.array(image2.convert('L'))
    score, _ = ssim(image1, image2, full=True)
    return score
