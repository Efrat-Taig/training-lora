# LoRA Image Generation Script
# Author: Efrat Taig
#
# This script demonstrates how to use a pre-trained diffusion model (BRIA-2.3)
# with LoRA (Low-Rank Adaptation) weights to generate an image based on a text prompt.
# It loads the model, applies LoRA weights, generates an image, and saves it to disk.
#
# Key steps:
# 1. Import required libraries
# 2. Load the pre-trained model
# 3. Apply LoRA weights
# 4. Generate an image from a text prompt
# 5. Save the generated image

from diffusers import DiffusionPipeline
import torch
from safetensors.torch import load_file

# Define model and LoRA parameters
model_id = "briaai/BRIA-2.3"
checkpoint_num = 100000
lora_model_path = f'lora/checkpoint-{checkpoint_num}/pytorch_lora_weights.safetensors'

# Load the pre-trained model
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Move the model to GPU for faster processing

# Load and apply LoRA weights
state_dict = load_file(lora_model_path)
pipe.unet.load_attn_procs(state_dict)

# Define the prompt and generation parameters
prompt =  "A distant, tranquil view of the ocean at sunrise, seen from a high-rise window with soft, warm reflections. Aesthetic background for profile picture.",

guidance_scale = 1

# Generate the image
image = pipe(prompt, num_inference_steps=30, guidance_scale=guidance_scale).images[0]

# Define the path where the image will be saved
image_path = f'lora/output/{prompt}.png'

# Save the generated image
image.save(image_path)

print(f"Image saved to: {image_path}").py
