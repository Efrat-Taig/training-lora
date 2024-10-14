# Lora_Multi-Checkpoint_Image_Generation_Comparer.py
# Author: Efrat Taig
#
# This script generates and compares images using different LoRA (Low-Rank Adaptation) 
# checkpoints for a pre-trained diffusion model. It creates a grid of images for visual 
# comparison, showing how the model's output changes across different checkpoints and seeds.
#
# Key features:
# - Generates images with and without LoRA
# - Creates comparison grids for multiple prompts
# - Saves individual images and final comparison grids
# - Supports custom seeds and multiple checkpoints

import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from PIL import Image, ImageDraw, ImageFont
import os
import re
import random

def create_image(pipe, prompt, seed, guidance_scale=1, num_inference_steps=30):
    """
    Generate an image using the provided diffusion pipeline.
    
    :param pipe: The diffusion pipeline
    :param prompt: Text prompt for image generation
    :param seed: Random seed for reproducibility
    :param guidance_scale: Controls the influence of the text prompt
    :param num_inference_steps: Number of denoising steps
    :return: Generated image
    """
    generator = torch.Generator("cuda").manual_seed(seed)
    return pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]

def create_comparison_grid(images, seeds, checkpoint_nums, target_size=(4000, 4000)):
    """
    Create a grid of images for comparison across different checkpoints and seeds.
    
    :param images: List of lists containing images
    :param seeds: List of seeds used for generation
    :param checkpoint_nums: List of checkpoint numbers
    :param target_size: Target size for the grid image
    :return: PIL Image object of the comparison grid
    """
    num_rows = len(images)
    num_cols = len(images[0])
    
    # Fixed target size
    grid_width, grid_height = 4000, 4000
    
    # Calculate cell size
    cell_width = grid_width // num_cols
    cell_height = (grid_height - 100) // (num_rows + 1)  # 100 pixels for the header, +1 for the header row
    
    # Create a new image with white background
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid_image)
    font = ImageFont.load_default()
    
    # Add column labels (checkpoint numbers)
    for j, checkpoint in enumerate(['No LoRA'] + checkpoint_nums):
        label = f'Checkpoint: {checkpoint}'
        draw.text((j * cell_width + 5, 5), label, fill='black', font=font)
    
    # Paste and resize images, add row labels (seeds)
    for i, row in enumerate(images):
        for j, img in enumerate(row):
            resized_img = img.resize((cell_width, cell_height), Image.LANCZOS)
            grid_image.paste(resized_img, (j * cell_width, (i + 1) * cell_height + 100))
        
        # Add seed label
        label = f'Seed: {seeds[i]}'
        draw.text((5, (i + 1) * cell_height + 105), label, fill='black', font=font)
    
    return grid_image

def sanitize_filename(filename):
    """
    Remove special characters from filename and truncate if necessary.
    
    :param filename: Original filename
    :return: Sanitized filename
    """
    filename = re.sub(r"[^\w\-\s]", "", filename)
    filename = filename.replace(" ", "_")
    return filename[:50]

def save_individual_image(image, main_output_dir, checkpoint, prompt, seed):
    """
    Save an individual generated image.
    
    :param image: PIL Image object
    :param main_output_dir: Main output directory
    :param checkpoint: Checkpoint number or 'no_lora'
    :param prompt: Text prompt used for generation
    :param seed: Seed used for generation
    """
    checkpoint_dir = os.path.join(main_output_dir, f'checkpoint_{checkpoint}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    sanitized_prompt = sanitize_filename(prompt)
    filename = f'{sanitized_prompt}_seed_{seed}.png'
    output_path = os.path.join(checkpoint_dir, filename)
    image.save(output_path)

def process_checkpoints(model_id, dataset_name, checkpoint_nums, lora_model_paths, prompts, main_output_dir, num_seeds=5):
    """
    Process multiple checkpoints, generate images, and create comparison grids.
    
    :param model_id: ID of the pre-trained model
    :param dataset_name: Name of the dataset (used for output naming)
    :param checkpoint_nums: List of checkpoint numbers
    :param lora_model_paths: List of paths to LoRA model weights
    :param prompts: List of text prompts for image generation
    :param main_output_dir: Main output directory
    :param num_seeds: Number of random seeds to use
    """
    seeds = [random.randint(0, 1000000) for _ in range(num_seeds)]
    seeds[0] = 0  # Set first seed to 0 for consistency
    # Initialize the grid for each prompt
    grids = {prompt: [] for prompt in prompts}
    
    # Create pipeline without LoRA
    print("Creating images without LoRA")
    pipe_no_lora = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    
    for prompt in prompts:
        print(f"\nProcessing prompt: {prompt}")
        no_lora_images = []
        for seed in seeds:
            print(f"  Generating image for seed: {seed}")
            image = create_image(pipe_no_lora, prompt, seed)
            no_lora_images.append(image)
            save_individual_image(image, main_output_dir, 'no_lora', prompt, seed)
        grids[prompt] = [no_lora_images]
    
    # Free up GPU memory
    del pipe_no_lora
    torch.cuda.empty_cache()
    
    # Process each LoRA checkpoint
    for idx, (checkpoint_num, lora_model_path) in enumerate(zip(checkpoint_nums, lora_model_paths)):
        print(f"\nProcessing checkpoint: {checkpoint_num}")
        
        # Create pipeline with LoRA
        pipe_with_lora = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        state_dict = load_file(lora_model_path)
        pipe_with_lora.unet.load_attn_procs(state_dict)
        
        for prompt in prompts:
            print(f"Processing prompt: {prompt}")
            lora_images = []
            for seed in seeds:
                print(f"  Generating image for seed: {seed}")
                image = create_image(pipe_with_lora, prompt, seed)
                lora_images.append(image)
                save_individual_image(image, main_output_dir, checkpoint_num, prompt, seed)
            grids[prompt].append(lora_images)
        
        # Free up GPU memory
        del pipe_with_lora
        torch.cuda.empty_cache()
    
    # Create and save the final comparison grids
    for i, (prompt, grid_images) in enumerate(grids.items()):
        final_grid = create_comparison_grid(list(zip(*grid_images)), seeds, checkpoint_nums)
        sanitized_prompt = sanitize_filename(prompt)
        final_filename = f'final_comparison_grid_{i+1}_{sanitized_prompt}.png'
        final_output_path = os.path.join(main_output_dir, final_filename)
        final_grid.save(final_output_path)
        print(f"Final comparison grid saved: {final_filename}")

    print("\nAll prompts processed successfully!")

# Main execution
model_id = "briaai/BRIA-2.3"
dataset_name = "Modern_Blurred_SeaView"

checkpoint_nums = [100,500,1400]

lora_model_paths = [
    f'lora/output/{dataset_name}/checkpoint-{num}/pytorch_lora_weights.safetensors'
    for num in checkpoint_nums
]

prompts = [
 "A distant, tranquil view of the ocean at sunrise, seen from a high-rise window with soft, warm reflections. Aesthetic background for profile picture.",
 "A softly blurred perspective of endless waves reaching the horizon, viewed from a tall building on a misty morning. Aesthetic background for profile picture.",
 "A calm, expansive ocean view with subtle reflections, as seen through a modern building's glass window on a hazy day. Aesthetic background for profile picture.",
]

main_output_dir = f'lora/output_bria/results_{dataset_name}'

os.makedirs(main_output_dir, exist_ok=True)

num_seeds = 3  # Number of fixed seeds to be created

# Run the main processing function
process_checkpoints(model_id, dataset_name, checkpoint_nums, lora_model_paths, prompts, main_output_dir, num_seeds)
