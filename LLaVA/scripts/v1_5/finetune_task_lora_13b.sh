#!/bin/bash
export PYTHONPATH=./:$PYTHONPATH
datapaths=(
    # "2000_llava_med_filtered_10k_5000_filtered_llava_v1_5_mix665k"
)
port=29500

for data_path in "${datapaths[@]}"; do
    echo "using data file: $data_path"
    parent_dir="../trial_dataset/"
    full_data_path="$parent_dir/$data_path.json"
    output_dir="../../data/output_model/lora/llava-v1.5-13b-$data_path-lora"

    suitable_gpus=()
    while [ ${#suitable_gpus[@]} -lt 4 ]; do
        suitable_gpus=()
        for gpu_id in $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits); do
            if [ ${#suitable_gpus[@]} -eq 4 ]; then
                break
            fi
            mem_util=$(nvidia-smi --id=$gpu_id --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{print ($1/$2)*100}')
            gpu_util=$(nvidia-smi --id=$gpu_id --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{print $1}')
            if (( $(echo "$mem_util < 5" | bc -l) )) && (( $(echo "$gpu_util < 5" | bc -l) )); then
                suitable_gpus+=($gpu_id)
            fi
        done
        if [ ${#suitable_gpus[@]} -lt 4 ]; then
            echo "No sufficient gpus. Retry in 5 seconds..."
            sleep 5
        fi
    done

    echo "${suitable_gpus[@]} GPU are found! executing the experiment on these devices!"
    # suitable_gpus=(0 1 6 7)
    gpu_include=$(printf ",%s" "${suitable_gpus[@]}")
    gpu_include=${gpu_include:1} 
    gpu_include="localhost:${gpu_include}"

    deepspeed --master_port $port \
        --include $gpu_include  llava/train/train_mem.py \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --deepspeed ./scripts/zero3_offload.json \
        --model_name_or_path "../../data/output_model/lora/llava-v1.5-13b" \
        --version v1 \
        --data_path $full_data_path \
        --image_folder ./playground/data \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir $output_dir \
        --num_train_epochs 3 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy steps \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 8 \
        --lazy_preprocess True \
        --report_to wandb && \

    port=$((port+1)) \

done