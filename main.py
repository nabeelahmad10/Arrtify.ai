import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import clip
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import io

# Optimized Configurations
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 20  # Reduced steps to speed up generation
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (512, 512)  # Reduced resolution for faster processing
    image_gen_guidance_scale = 7  # Adjust scale to balance speed and quality

# Load Stable Diffusion model with mixed precision (FP16) if available
@st.cache_resource
def load_sd_model():
    return StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id,
        torch_dtype=torch.float16 if CFG.device == "cuda" else torch.float32,
        revision="fp16" if CFG.device == "cuda" else None
    ).to(CFG.device)

# Load CLIP model and preprocess function
@st.cache_resource
def load_clip_model():
    return clip.load("ViT-B/32", device=CFG.device)

image_gen_model = load_sd_model()
clip_model, preprocess = load_clip_model()

# Generate Image Function
def generate_image(prompt):
    image = image_gen_model(
        prompt,
        num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    return image

# CLIP score calculation (no change, still optimized)
def calculate_clip_score(image1, image2, text):
    text_inputs = clip.tokenize([text]).to(CFG.device)
    image1_tensor = preprocess(image1).unsqueeze(0).to(CFG.device)
    image2_tensor = preprocess(image2).unsqueeze(0).to(CFG.device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        image1_features = clip_model.encode_image(image1_tensor)
        image2_features = clip_model.encode_image(image2_tensor)

        similarity1 = torch.cosine_similarity(text_features, image1_features)
        similarity2 = torch.cosine_similarity(text_features, image2_features)

    return similarity1.item(), similarity2.item()

# SSIM calculation (no change, still optimized)
def calculate_ssim(image1, image2):
    image1_gray = np.array(image1.convert("L"))
    image2_gray = np.array(image2.convert("L"))
    ssim_value = ssim(image1_gray, image2_gray)
    return ssim_value

# Streamlit UI
st.title("Optimized Text-to-Image Generator and Comparison")

# Select between generate or compare images
option = st.selectbox("Choose an option", ["Generate Image", "Compare Images"])

if option == "Generate Image":
    prompt = st.text_input("Enter a prompt to generate an image:")
    if st.button("Generate"):
        if prompt:
            with st.spinner('Generating image...'):
                generated_image = generate_image(prompt)
            st.image(generated_image, caption="Generated Image", use_column_width=True)

elif option == "Compare Images":
    prompt = st.text_input("Enter a prompt for comparison:")
    uploaded_image = st.file_uploader("Upload an image to compare", type=["png", "jpg", "jpeg"])

    if st.button("Compare") and prompt and uploaded_image:
        with st.spinner('Generating and comparing images...'):
            generated_image = generate_image(prompt)
            comparison_image = Image.open(uploaded_image)

            # Calculate CLIP and SSIM scores
            clip_score1, clip_score2 = calculate_clip_score(generated_image, comparison_image, prompt)
            ssim_value = calculate_ssim(generated_image, comparison_image)

        # Display results
        st.image(generated_image, caption="Generated Image", use_column_width=True)
        st.image(comparison_image, caption="Uploaded Image", use_column_width=True)
        st.write(f"CLIP Similarity to Generated Image: {clip_score1:.4f}")
        st.write(f"CLIP Similarity to Uploaded Image: {clip_score2:.4f}")
        st.write(f"Structural Similarity Index (SSIM): {ssim_value:.4f}")
