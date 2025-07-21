lora_rank=16
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
MAX_STEPS=5700
SAVE_STEPS=5700
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
model_name_or_path="chatglm3-6b"
your_data_path="datasets"
your_checkpopint_path="saved/moelora"
MAX_SOURCE_LENGTH=2048

peft_path=""

Training Command
deepspeed --num_gpus=1 --master_port $MASTER_PORT run_mlora.py \
    --deepspeed src/ds.config \
    --do_train \
    --train_file $your_data_path/train_all4true_tasks.json \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 512 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps ${MAX_STEPS} \
    --logging_steps 100 \
    --save_steps ${SAVE_STEPS} \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --fp16 \
    --lora_name moelora \
    --expert_num 4


# export CUDA_VISIBLE_DEVICES=0
deepspeed --num_gpus=1 --master_port $MASTER_PORT run_mlora.py \
    --do_predict \
    --test_file $your_data_path/test_all4true_tasks.json \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --peft_path $your_checkpopint_path/checkpoint-${MAX_STEPS} \
    --output_dir results/pred/moelora \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 512 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \
    --lora_name moelora \
    --expert_num 4