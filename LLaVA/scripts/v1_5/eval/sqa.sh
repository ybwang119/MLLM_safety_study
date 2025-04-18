#!/bin/bash
model_names=(
    "llava-v1.5-7b" 
    "llava-v1.5-7b-1999_vlguard_train_llava_format_5000_textvqa_20k"
    "llava-v1.5-7b-2000_vlguard_train_llava_format_5000_llava_v1_5_mix665k"
    "llava-v1.5-7b-2000_vlguard_train_llava_format_5000_llava_v1_5_mix665k_one_turn"
    "llava-v1.5-7b-2977_one_turn_vlguard_5000_textvqa_20k"
    "llava-v1.5-7b-Mixed"
    "llava-v1.5-7b-modified_2977_one_turn_vlguard_5000_textvqa_20k"
    )
parent_dir="../ckpts/"
command1s=()
command2s=()
for model_name in "${model_names[@]}"; do
    eval_command_1="python -m llava.eval.model_vqa_science --model-path $parent_dir$model_name --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json --image-folder ./playground/data/eval/scienceqa/images/test --answers-file ./playground/data/eval/scienceqa/answers/$model_name.jsonl --single-pred-prompt --temperature 0 --conv-mode vicuna_v1"
    eval_command_2="python llava/eval/eval_science_qa.py --base-dir ./playground/data/eval/scienceqa --result-file ./playground/data/eval/scienceqa/answers/${model_name}.jsonl --output-file ./playground/data/eval/scienceqa/answers/${model_name}_output.jsonl --output-result ./playground/data/eval/scienceqa/answers/${model_name}_result.json"
    command1s+=("$eval_command_1")
    command2s+=("$eval_command_2")
done
echo "开始测试！"

# 定义一个函数来运行命令并等待其完成
run_command() {
    local cmd=$1
    local gpu_id=$2
    echo "Running command on GPU $gpu_id: $cmd"
    # 在这里添加实际运行命令的代码，例如：
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd
}

# 定义一个函数来检查 GPU 利用率
check_gpu_usage() {
    local gpu_id=$1
    local usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id | tr -d ' ')
    usage=$(echo $usage | tr -d '[:space:]')
    if [ "$usage" -lt 60 ]; then
        return 0
    else
        return 1
    fi
}
# 定义一个函数来检查 GPU 显存利用率
check_gpu_memory_usage() {
    local gpu_id=$1
    local memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id | tr -d ' ')
    local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpu_id | tr -d ' ')
    memory_usage=$(echo $memory_usage | tr -d '[:space:]')
    memory_total=$(echo $memory_total | tr -d '[:space:]')
    local memory_utilization=$((memory_usage * 100 / memory_total))
    if [ "$memory_utilization" -lt 50 ]; then
        return 0
    else
        return 1
    fi
}

gpu_ids=($1)
gpu_count=${#gpu_ids[@]}

# 定义一个数组来记录每个 GPU 的等待时间
declare -A gpu_wait_time
for gpu_id in "${gpu_ids[@]}"; do
    gpu_wait_time[$gpu_id]=0
done


# 循环遍历命令数组并轮流运行
# for cmd in "${command1s[@]}"; do
#     while true; do
#         current_time=$(date +%s)
#         for gpu_id in "${gpu_ids[@]}"; do
#             if check_gpu_usage $gpu_id && [ $current_time -ge ${gpu_wait_time[$gpu_id]} ]; then
#                 run_command "$cmd" $gpu_id &
#                 gpu_wait_time[$gpu_id]=$((current_time + 5))  # 设置 GPU 的等待时间
#                 break 2  # 跳出两个循环
#             fi
#         done
#         sleep 60  # 等待一秒钟后重试
#     done
# done

for cmd in "${command2s[@]}"; do
echo "$cmd"
    $cmd
done
wait