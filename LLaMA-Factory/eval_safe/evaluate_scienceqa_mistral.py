import argparse
import os
from tqdm import tqdm
# os.environ['CUDA_VISIBLE_DEVICES']="0"
import torch
import json
import shortuuid
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import math
from PIL import Image
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    model = LlavaNextForConditionalGeneration.from_pretrained(args.model_base,torch_dtype=torch.bfloat16,device_map="auto").eval()
    peft_model_id = args.model_path
    try:    
        model.load_adapter(peft_model_id)    
        print(f"Successfully load lora weights to {args.model_base} from {args.model_path}!")

    except:
        print(f"model_base is {args.model_base} while model_path is {args.model_path}! check weather the aim is to evaluate base model or fine-tuned model!")
    processor = LlavaNextProcessor.from_pretrained(args.model_base)
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):        
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = "<image>"+qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
            if args.single_pred_prompt:
                qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            conversation = [
            {"role": "user", "content": [{"type": "text", "text": qs},{"type": "image"}]},
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        else:
            image=None
            if args.single_pred_prompt:
                qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            conversation = [
            {"role": "user", "content": [{"type": "text", "text": qs}]},
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=200,temperature=args.temperature,pad_token_id=processor.tokenizer.eos_token_id)
            response=processor.decode(output[0], skip_special_tokens=True).split("[/INST] ")[-1].strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": response,
                                   "answer_id": ans_id,
                                   "model_id": peft_model_id,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default="huggingface/hub/models--llava-hf--llava-v1.6-mistral-7b-hf/xxxxxxxx")
    parser.add_argument("--model-path", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--image-folder", type=str, default="../LLaVA/playground/data/eval/scienceqa/images/test")
    parser.add_argument("--question-file", type=str, default="../LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json")
    parser.add_argument("--answers-file", type=str, default="../scienceqa_llava/llava-v1.6-mistral-7b.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)
