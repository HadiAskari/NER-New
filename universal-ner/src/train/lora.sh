MODEL=Universal-NER/UniNER-7B-all
DATA=./vulnerability_train.json

mkdir -p saved_models

deepspeed --include localhost:1,2,3 ./fastchat/train/train_lora.py \
    --model_name_or_path ${MODEL} \
    --data_path ${DATA} \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --output_dir saved_models/vulnerability \
    --model_max_length 2048 \
    --q_lora False \
    --deepspeed playground/deepspeed_config_s2.json \
    --gradient_checkpointing True \
    --lazy_preprocess True

    