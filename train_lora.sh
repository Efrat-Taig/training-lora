#   based on this:  https://huggingface.co/docs/diffusers/en/training/lora


export MODEL_NAME="briaai/BRIA-2.3"
export DATASET_NAME="Negev900/Modern_Blurred_SeaView"


accelerate launch --mixed_precision="bf16"  train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=1 \
  --resolution=1024 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --rank=32 \
  --allow_tf32 \
  --use_8bit_adam \
  --gradient_accumulation_steps=6 \
  --max_train_steps=800 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=./output_bria/Modern_Blurred_SeaView \
  --checkpointing_steps=100 \
  --validation_prompt="" \
  --num_validation_images=1 \
  --validation_epochs=1000000 \
  --seed=1337 \
  --image_column="image" \
  --caption_column="prompt"



#   can add : --resume_from_checkpoint=./output_bria/model_name/checkpoint-500 \

#   Must have: 
#  wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora_sdxl.py
#  pip install -U transformers accelerate
