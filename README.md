
# Training LoRA with Bria's Text-to-Image Model

This repository contains files for training LoRA using Bria's compatible text-to-image models with SDXL, allowing easy setup and execution.

## Files Included
- **[Train_text_to_image_lora_sdxl.py](https://github.com/Efrat-Taig/training-lora/blob/main/train_text_to_image_lora_sdxl.py)**: Core training script for LoRA, adapted from Hugging Face Diffusers.
- **[Train_lora.sh](https://github.com/Efrat-Taig/training-lora/blob/main/train_lora.sh)**: Script to configure and initiate LoRA training.

## Getting Started

1. **Run the Training Script**:
   Execute `Train_lora.sh` to set up configurations and run the `Train_text_to_image_lora_sdxl.p` script.

2. **Bria Model Compatibility**:
   Bria's text-to-image model (Bria 2.3) is fully compatible with the SDXL framework, enabling you to train LoRA on Bria 2.3 just as with SDXL models.


 Bria's models are designed to be compatible with [Diffusers](https://github.com/huggingface/diffusers/tree/main), specifically model version [Bria 2.3](https://huggingface.co/briaai/BRIA-2.3), is compatible with SDXL. This means you can train LoRA on  [Bria 2.3](https://huggingface.co/briaai/BRIA-2.3) in the same way as you would with SDXL models.

## Data Preparation

1. **Data Generation**:
   - Generate your dataset and upload it to [Hugging Face](https://huggingface.co/) to streamline the import process.
   - Example dataset: [Modern Blurred SeaView](https://huggingface.co/datasets/Negev900/Modern_Blurred_SeaView)

2. **Detailed Guide**:
   For step-by-step instructions on preparing your dataset, refer to [this article](link-to-article).

## Results

The repository also includes sample result images to showcase the LoRA training output.

