import argparse
import os
from tqdm import tqdm
# os.environ['CUDA_VISIBLE_DEVICES']="1"
import torch
from torch import nn
import json
import shortuuid
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaConfig
import transformers
import numpy as np
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

class LlavaMultiModalProjectorYiVL(nn.Module):
    def __init__(self, config: "LlavaConfig"):
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_2 = nn.LayerNorm(config.text_config.hidden_size, bias=True)
        self.linear_3 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_4 = nn.LayerNorm(config.text_config.hidden_size, bias=True)
        self.act = nn.GELU()

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_3(hidden_states)
        hidden_states = self.linear_4(hidden_states)
        return hidden_states

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    transformers.models.llava.modeling_llava.LlavaMultiModalProjector = LlavaMultiModalProjectorYiVL
    model_id = args.model_base
    model = AutoModelForVision2Seq.from_pretrained(model_id)
    peft_model_id = args.model_path
    if peft_model_id.split('/')[-1] != "Yi-VL-6B":
        model.load_adapter(peft_model_id)    
    model=model.to(args.device)
    processor = AutoProcessor.from_pretrained(model_id)
    model = model.eval()

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):        
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = [Image.open(os.path.join(args.image_folder, image_file))]
            cur_prompt = "<image>\n"+ cur_prompt
        else:
            image=None
        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt =cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
        messages = [
        { "role": "user", "content": f"{cur_prompt}" }
        ]
        text = [processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)]
        with torch.inference_mode():
            inputs = processor(text=text, images=image, return_tensors='pt').to(model.device, torch.float16)
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=args.temperature,
                use_cache=True
                )
            outputs = processor.batch_decode(output, skip_special_tokens=True)[0].split("Assistant: ")[-1].strip() 

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": peft_model_id,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default="BUAADreamer/Yi-VL-6B-hf")
    parser.add_argument("--model-path", type=str, default="./lora/Yi-VL-6B-my1")
    parser.add_argument("--image-folder", type=str, default="LLaVA/playground/data/eval/scienceqa/images/test")
    parser.add_argument("--question-file", type=str, default="LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json")
    parser.add_argument("--answers-file", type=str, default="scienceqa_minigptv2/answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)
