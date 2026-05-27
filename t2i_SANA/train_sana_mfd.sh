export MODEL_NAME="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
export PROMPT_DATASET_DIR="datasets/labels_cleaned.tsv"
export OUTPUT_DIR="outputs"
export NUM_GPUS=8
export GPU_IDS="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

accelerate launch \
  --num_processes="${NUM_GPUS}" \
  --num_machines=1 \
  scripts/train_sana_mfd.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --prompt_dataset=$PROMPT_DATASET_DIR \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --rank=64 \
  --lora_alpha=64 \
  --train_batch_size=1 \
  --use_8bit_adam \
  --learning_rate_stu=1e-4 \
  --learning_rate_aux=1e-4 \
  --ode_step_size=0.02 \
  --aux_steps_per_cycle=50 \
  --student_steps_per_cycle=20 \
  --r_not_equal_t_ratio=1.0 \
  --optimizer="AdamW" \
  --max_grad_norm=1.0 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=30000 \
  --validation_prompt="a bear riding a skateboard, 4k, high resolution" \
  --checkpointing_steps=1000 \
  --validation_epochs=25 \
  --validation_steps=500 \
  --val_resolution=1024 \
  --num_inference_steps=4 \
  --adam_beta1=0.9 \
  --adam_beta2=0.999 \
  --adam_weight_decay=1e-4 \
  --seed="42" 
#   --gradient_checkpointing