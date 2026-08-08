cd src/open-r1-multimodal

RUN_NAME="test-kadid"
export LOG_PATH="./log_$RUN_NAME.txt"

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_jsonl.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path "<MODEL_NAME_OR_PATH>" \
    --question_template scoring \
    --dataset_name KADID-10K \
    --image_folders "<IMAGE_FOLDER>" \
    --gray_image_folders "<GRAY_IMAGE_FOLDER>" \
    --data_file_paths "<DATA_FILE_PATH>" \
    --freeze_vision_modules false \
    --max_prompt_length 1024 \
    --num_generations 6 \
    --per_device_train_batch_size 42 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 5 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true
