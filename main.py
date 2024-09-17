import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import os
import base64

# Optimized Configurations
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 15  # Reduced steps to speed up generation
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (384, 384)  # Reduced resolution for faster processing
    image_gen_guidance_scale = 7  # Adjust scale to balance speed and quality

# Load Stable Diffusion model with mixed precision (FP16) if available
@st.cache_resource
def load_sd_model():
    return StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id,
        torch_dtype=torch.float16 if CFG.device == "cuda" else torch.float32,
        revision="fp16" if CFG.device == "cuda" else None
    ).to(CFG.device)

image_gen_model = load_sd_model()

# Generate Image Function
def generate_image(prompt):
    image = image_gen_model(
        prompt,
        num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    return image

# Function to convert image to base64 string
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Streamlit UI
st.title("Optimized Text-to-Image Generator")

# Convert background image to base64
background_image_path = "static/Artifyai.jpg"  # Correct path to your JPG file
if os.path.exists(background_image_path):
    base64_background = image_to_base64(background_image_path)
    # Apply the background using base64 encoding
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_background}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Background image not found. Make sure the JPG file is in the 'static' folder.")

# Text input for the image prompt
prompt = st.text_input("Enter a prompt to generate an image:")

if st.button("Generate"):
    if prompt:
        with st.spinner('Generating image...'):
            generated_image = generate_image(prompt)
        st.image(generated_image, caption="Generated Image", use_column_width=True)

        # Provide option to download image
        img_byte_arr = io.BytesIO()
        generated_image.save(img_byte_arr, format='PNG')  # Convert image to bytes
        img_byte_arr = img_byte_arr.getvalue()  # Get image bytes

        # Download button for the generated image
        st.download_button(
            label="Download Image",
            data=img_byte_arr,
            file_name="generated_image.png",
            mime="image/png"
        )
