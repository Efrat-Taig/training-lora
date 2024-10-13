
# Training LoRA with Bria's Text-to-Image Model

This repository contains files for training LoRA using Bria's compatible text-to-image models with SDXL, allowing easy setup and execution.

### Files Included:
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

Sample from the Modern Blurred SeaView Dataset:

<img src="https://github.com/Efrat-Taig/training-lora/blob/main/Data_set_sample.png" width="600">>


2. **Detailed Guide**:
   For step-by-step instructions on preparing your dataset, refer to [this article](link-to-article).

## LoRA Training Results Sample Collection:
In the images, each column represents a different model variant after LoRA fine-tuning, while each row corresponds to a different seed. The prompt used to generate each image is noted below the images. Notably,
at a checkpoint of 800 steps, the generated data begins to closely resemble the training data, indicating effective adaptation of the model to the dataset characteristics. This demonstrates the model's ability to capture and replicate specific features of the training data within relatively few training steps.


<img src="https://github.com/Efrat-Taig/training-lora/blob/main/lora_res_2.png" width="600">>

<img src="https://github.com/Efrat-Taig/training-lora/blob/main/lora_res_1.png" width="600">>


