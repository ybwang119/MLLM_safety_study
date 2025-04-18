
# export HF_HOME=xxxxx/huggingface
# export HF_HUB_CACHE=xxxxx/huggingface/hub
gpu_nums=4



datapaths=(
    # "name of the dataset file"
)
port=29500
total_batch_size=128
train_batch_size=8  

exclude_gpus=(3)
for data_path in "${datapaths[@]}"; do
    parent_dir="data/"
    full_data_path="${data_path}_tr"
    output_dir="../../data/output_model/lora/llava-v1.6-mistral-7b-$data_path-lora"

    suitable_gpus=()
    while [ ${#suitable_gpus[@]} -lt $gpu_nums ]; do
        suitable_gpus=()
        for gpu_id in $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits); do
            if [[ " ${exclude_gpus[@]} " =~ " ${gpu_id} " ]]; then
                    continue
            fi
            if [ ${#suitable_gpus[@]} -eq $gpu_nums ]; then
                break
            fi
            mem_util=$(nvidia-smi --id=$gpu_id --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{print ($1/$2)*100}')
            gpu_util=$(nvidia-smi --id=$gpu_id --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{print $1}')
            if (( $(echo "$mem_util < 5" | bc -l) )) && (( $(echo "$gpu_util < 5" | bc -l) )); then
                suitable_gpus+=($gpu_id)
            fi
        done
        if [ ${#suitable_gpus[@]} -lt $gpu_nums ]; then
            echo "No sufficient gpus. Retry in 5 seconds..."
            sleep 5
        fi
    done

    echo "${suitable_gpus[@]} GPU are found! executing the experiment on these devices!"
    # suitable_gpus=(0 1 6 7)
    CUDA_VISIBLE_DEVICES=$(printf ",%s" "${suitable_gpus[@]}")
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:1}

    torchrun --master_port $port  --nproc_per_node=$gpu_nums src/train.py \
        --deepspeed examples/deepspeed/ds_z3_config.json \
        --lora_rank 128 \
        --lora_alpha 256 \
        --stage sft \
        --do_train \
        --max_samples 100000 \
        --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
        --freeze_vision_tower true \
        --preprocessing_num_workers 8 \
        --dataset $full_data_path \
        --template llava_next_mistral \
        --finetuning_type lora \
        --lora_target all \
        --output_dir $output_dir \
        --overwrite_cache \
        --overwrite_output_dir true \
        --warmup_ratio 0.03 \
        --weight_decay 0. \
        --per_device_train_batch_size $train_batch_size \
        --gradient_accumulation_steps $(($total_batch_size/ $train_batch_size / $gpu_nums)) \
        --ddp_timeout 9000 \
        --learning_rate 2e-4 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --cutoff_len 2048 \
        --save_steps 50000 \
        --plot_loss \
        --num_train_epochs 3.0 \
        --bf16 \
        --save_only_model true \
        --report_to wandb \
        --run_name mistral-7b-$data_path
done