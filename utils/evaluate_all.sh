#!/bin/bash
conda init bash
source ~/.bashrc
conda activate VLM

cd ../MLLM_safety_study/LLaVA
export PYTHONPATH=../MLLM_safety_study/LLaVA/:$PYTHONPATH
cd ../
echo "start testing!"
run_command() {
    local cmd=$1
    local gpu_id=$2
    echo "Running command on GPU $gpu_id: $cmd"

    CUDA_VISIBLE_DEVICES=$gpu_id $cmd
}


check_gpu_usage() {
    local gpu_id=$1
    local usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id | tr -d ' ')
    usage=$(echo $usage | tr -d '[:space:]')
    if [ "$usage" -lt 30 ]; then
        return 0
    else
        return 1
    fi
}

check_gpu_memory_usage() {
    local gpu_id=$1
    local memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id | tr -d ' ')
    local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpu_id | tr -d ' ')
    memory_usage=$(echo $memory_usage | tr -d '[:space:]')
    memory_total=$(echo $memory_total | tr -d '[:space:]')
    local memory_utilization=$((memory_usage * 100 / memory_total))
    if [ "$memory_utilization" -lt 10 ]; then
        return 0
    else
        return 1
    fi
}

gpu_ids=($1)
gpu_count=${#gpu_ids[@]}

declare -A gpu_wait_time
for gpu_id in "${gpu_ids[@]}"; do
    gpu_wait_time[$gpu_id]=0
done

model_names=(
   # your data file here
    )
parent_dir="./lora/"
input_mode="mix"
model_base="xxx/llava-v1.5-7b"

commands=()
for model_name in "${model_names[@]}"; do
    eval_command="python evaluate_vizwiz.py --model-path $parent_dir$model_name --model-base $model_base "
    commands+=("$eval_command")
done


#xstest
for model_name in "${model_names[@]}"; do
    eval_command="python evaluate_jsonfile.py --model-path $parent_dir$model_name --model-base $model_base --dataset_file ../data/dataset/json_file/xstest.json --output_file_path ./xstest_result"
    commands+=("$eval_command")
done

##figstep
for model_name in "${model_names[@]}"; do
    eval_command="python evaluate_figstep_llava.py --model-path $parent_dir$model_name --model-base $model_base"
    commands+=("$eval_command")
done

for model_name in "${model_names[@]}"; do
    for i in $(seq 0 12); do
    eval_command="python evaluate_MM_llava_ablation.py --model-path $parent_dir$model_name --model-base $model_base --input-mode $input_mode --index $i"
    commands+=("$eval_command")
    done
done

# && check_gpu_memory_usage $gpu_id
for cmd in "${commands[@]}"; do
    while true; do
        current_time=$(date +%s)
        for gpu_id in "${gpu_ids[@]}"; do
            if check_gpu_usage $gpu_id  && check_gpu_memory_usage $gpu_id && [ $current_time -ge ${gpu_wait_time[$gpu_id]} ]; then
                run_command "$cmd" $gpu_id &
                gpu_wait_time[$gpu_id]=$((current_time + 40)) 
                break 2  
            fi
        done
        sleep 1  
    done

done
wait
## evaluate_str
cd ../utils
commands=()
for model_name in "${model_names[@]}"; do
    eval_command="python judge_figstep_str.py --model_name $model_name"
    commands+=("$eval_command")
    eval_command="python judge_mmbench_str.py --model_name $model_name"
    commands+=("$eval_command")
    eval_command="python evaluate_xstest.py --model_name $model_name"
    commands+=("$eval_command")
done 


for cmd in "${commands[@]}"; do
    while true; do
        current_time=$(date +%s)
        for gpu_id in "${gpu_ids[@]}"; do
            if check_gpu_usage $gpu_id && check_gpu_memory_usage $gpu_id && [ $current_time -ge ${gpu_wait_time[$gpu_id]} ]; then
                run_command "$cmd" $gpu_id &
                gpu_wait_time[$gpu_id]=$((current_time)) 
                break 2  
            fi
        done
        sleep 1  
    done
done
wait

cd ../MLLM_safety_study/LLaVA
parent_dir="../lora/"

scienceqa_commands=()
for model_name in "${model_names[@]}"; do
    eval_command_1="python -m llava.eval.model_vqa_science --model-path $parent_dir$model_name --model-base $model_base --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json --image-folder ./playground/data/eval/scienceqa/images/test --answers-file ./playground/data/eval/scienceqa/answers/$model_name.jsonl --single-pred-prompt --temperature 0 --conv-mode vicuna_v1"
    scienceqa_commands+=("$eval_command_1")
done
for cmd in "${scienceqa_commands[@]}"; do
    while true; do
        current_time=$(date +%s)
        for gpu_id in "${gpu_ids[@]}"; do
            if check_gpu_usage $gpu_id && check_gpu_memory_usage $gpu_id && [ $current_time -ge ${gpu_wait_time[$gpu_id]} ]; then
                run_command "$cmd" $gpu_id &
                gpu_wait_time[$gpu_id]=$((current_time + 40))  
                break 2  
            fi
        done
        sleep 1  
    done
done
wait

scienceqa_commands=()
for model_name in "${model_names[@]}"; do
    eval_command_2="python llava/eval/eval_science_qa.py --base-dir ./playground/data/eval/scienceqa --result-file ./playground/data/eval/scienceqa/answers/${model_name}.jsonl --output-file ./playground/data/eval/scienceqa/answers/${model_name}_output.jsonl --output-result ./playground/data/eval/scienceqa/answers/${model_name}_result.json" 
    scienceqa_commands+=("$eval_command_2")
done

for cmd in "${scienceqa_commands[@]}"; do
    while true; do
        current_time=$(date +%s)
        for gpu_id in "${gpu_ids[@]}"; do
            if check_gpu_usage $gpu_id && check_gpu_memory_usage $gpu_id && [ $current_time -ge ${gpu_wait_time[$gpu_id]} ]; then
                run_command "$cmd" $gpu_id &
                gpu_wait_time[$gpu_id]=$((current_time))  
                break 2  
            fi
        done
        sleep 1  
    done
done
wait


cd xxxx/utils
judge_commands=()
for model_name in "${model_names[@]}"; do
    eval_command="python judge_from_llama_figstep.py --model_name $model_name"
    judge_commands+=("$eval_command")
done

for model_name in "${model_names[@]}"; do
    eval_command="python judge_from_llama_for_mmbench.py --model_name $model_name"
    judge_commands+=("$eval_command")
done

for cmd in "${judge_commands[@]}"; do
    while true; do
        current_time=$(date +%s)
        for gpu_id in "${gpu_ids[@]}"; do
            if check_gpu_usage $gpu_id && check_gpu_memory_usage $gpu_id && [ $current_time -ge ${gpu_wait_time[$gpu_id]} ]; then
                run_command "$cmd" $gpu_id &
                gpu_wait_time[$gpu_id]=$((current_time + 40))  
                break 2  
            fi
        done
        sleep 1 
    done
done
wait
