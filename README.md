
# Training LoRA with Bria's Text-to-Image Model

This repository contains files for training LoRA using Bria's  text-to-image foundation models, allowing easy setup and execution.

### Files Included:
- **[Train_text_to_image_lora_sdxl.py](https://github.com/Efrat-Taig/training-lora/blob/main/train_text_to_image_lora_sdxl.py)**: Core training script for LoRA, adapted from [Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py).
- **[Train_lora.sh](https://github.com/Efrat-Taig/training-lora/blob/main/train_lora.sh)**: Script to configure and initiate LoRA training.

# Getting Started

## To train Lora on [Bria 2.3](https://huggingface.co/briaai/BRIA-2.3) foundation model:

1. **Run the Training Script**:
   Execute `Train_lora.sh` to set up configurations and run the `Train_text_to_image_lora_sdxl.p` script.
   
     >💡 Note that using rank 128 consume lots of memory and increase model size, I recommend experimenting with lower ones e.g. 64, 32, 16


3. **Bria Model Compatibility**:
Bria's models are designed to be compatible with [Diffusers](https://github.com/huggingface/diffusers/tree/main), specifically model version [Bria 2.3](https://huggingface.co/briaai/BRIA-2.3), is compatible with SDXL. This means you can train LoRA on  [Bria 2.3](https://huggingface.co/briaai/BRIA-2.3) in the same way as you would with SDXL models.

     >💡 Note that you cannot connect a LoRA model trained on one foundation model type with a different foundation model type. For example, if you trained a LoRA on Bria's foundation model, it cannot be connected to SDXL, and vice versa. For more details on this, see my article [here](https://medium.com/@efrat_37973/replacing-backgrounds-with-diffusion-models-e86a4a96d9e9).


## Data Preparation

1. **Data Generation**:
   
Generate your dataset and store it using S3, local directories, or Hugging Face. I found [Hugging Face](https://huggingface.co) the most convenient, as it allows the data to flow smoothly into the training process. 
Below is an example dataset, Modern Blurred SeaView, I created; you can copy this concept to structure your own dataset similarly.

   - Generate your dataset and upload it to [Hugging Face](https://huggingface.co/) to streamline the import process.
   - Example dataset: [Modern Blurred SeaView](https://huggingface.co/datasets/Negev900/Modern_Blurred_SeaView)
   - A simple 2-minute tutorial on how to upload data to Hugging Face [here](https://www.youtube.com/watch?v=HaN6qCr_Afc&t=55s)

### Image Resolution Requirements

   - The optimal and required image resolution for this project is 1024x1024 pixels.
   - Images with lower resolutions will not work properly with the current setup.
   - Higher resolutions may consume excessive memory and are not supported at this time.

Note: Using the recommended 1024x1024 resolution ensures the best performance and results with our current model configuration.
     >💡Note that the more detailed part of saving the data is the JSONL file, which needs to be very precise. You can copy this structure from [here](https://huggingface.co/datasets/Negev900/Modern_Blurred_SeaView/blob/main/metadata.jsonl).


2. **Detailed Guide**:
   For step-by-step instructions on preparing your dataset, refer to [this article](link-to-article).

Sample from my [Modern Blurred SeaView](https://huggingface.co/datasets/Negev900/Modern_Blurred_SeaView) Dataset:

<img src="https://github.com/Efrat-Taig/training-lora/blob/main/Data_set_sample.png" width="600">>

## Evaluation Process:
### Evaluation Files Included:

1. **Single eval**:
   The  file [**BRIA_LoRA_Image_Generator.py**](https://github.com/Efrat-Taig/training-lora/blob/main/BRIA_LoRA_Image_Generator.py) generates results for a single checkpoint. 

2. **Multiple eval**:
   The  file [**Lora_Multi-Checkpoint_Image_Generation_Comparer.py**](https://github.com/Efrat-Taig/training-lora/blob/main/Lora_Multi-Checkpoint_Image_Generation_Comparer.py) loops through seeds, prompts, and checkpoints to generate multiple images, as shown in the examples below.
   
     >💡Note to customize the prompts and paths to fit your specific needs.
     >If you're having difficulty running the scrips, I suggest first running the inference for Bria 2.3 and ensuring all installations are set up as described in [this](https://huggingface.co/briaai/BRIA-2.3) model card.



## LoRA Training Results Sample Collection:

In the follwing images, each column represents a different model variant after LoRA fine-tuning, while each row corresponds to a different seed. The prompt used to generate each image is noted below the images. Notably,
at a checkpoint of 1400 steps, the generated data closely resemble the training data, indicating effective adaptation of the model to the dataset characteristics. This demonstrates the model's ability to capture and replicate specific features of the training data within relatively few training steps.


<img src="https://github.com/Efrat-Taig/training-lora/blob/main/lora_res_1.png" width="600">
<img src="https://github.com/Efrat-Taig/training-lora/blob/main/lora_res_2.png" width="600">
<img src="https://github.com/Efrat-Taig/training-lora/blob/main/lora_res_3.png" width="600">


## Troubleshooting

Here are some common issues and their solutions:

1. **CUDA out of memory error**
   - Try reducing the batch size 
   - Try reducing the rank size
   - Try reducing the batch image resolution
   - Ensure no other GPU-intensive tasks are running

2. **Module not found error**
   - Double-check that all required libraries are installed
   - Ensure you're using the correct Python environment

3. **Fine-tuning results in generated images**
   - Before integrating LoRA with background generation, ensure you're comfortable with the LoRA model's performance on its own
   - Experiment with different prompts to guide the image generation process
   - Try various seed values to explore different random initializations
   - Adjust the LoRA weight to control the influence of your trained style
   - Play with combinations of prompt, seed, and LoRA weight to achieve desired results
  
     
## Final Notes
For further assistance or collaboration opportunities, feel free to reach out:
- Email: efrat@bria.ai
- [LinkedIn](https://www.linkedin.com/in/efrattaig/)
- Join my [Discord community](https://discord.gg/Nxe9YW9zHS) for more information, tutorials, tools, and to connect with other users!


Academic users interested in accessing the model can [register here](https://docs.google.com/forms/d/1sSjxqS_2T4RB0dxnPjpygm7EXxa3RYNm2e4PUXQKnLo/edit) for access and further details.

For additional insights, refer to this [model information link](https://huggingface.co/briaai) or [article](https://medium.com/@efrat_37973/bridging-the-gap-from-academic-ai-to-ethical-business-models-89327517b940).
