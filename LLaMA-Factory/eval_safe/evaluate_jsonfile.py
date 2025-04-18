import argparse
import os
from tqdm.contrib import tzip
# os.environ['CUDA_VISIBLE_DEVICES']="1"
import torch
from torch import nn
import json
import requests
from PIL import Image
from io import BytesIO
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import transformers

class gt_dataset:
    def __init__(self, json_file):
        self.filedir = json_file
        dEFAULT_IMAGE_TOKEN = None
        with open(json_file, 'r') as f:
            self.raw_data = json.load(f)
        self.image=[i['image'] for i in self.raw_data]
        for i in ["<image>",'<image>\n']:
            if i in self.raw_data[0]['conversations'][0]['value']:
                dEFAULT_IMAGE_TOKEN=i
                break
        if dEFAULT_IMAGE_TOKEN is not None:
            self.user_prompt=[i['conversations'][0]['value'].split(dEFAULT_IMAGE_TOKEN)[-1] for i in self.raw_data if i['conversations'][0]['from']=='human']
        else:
            self.user_prompt=[i['conversations'][0]['value'] for i in self.raw_data if i['conversations'][0]['from']=='human']
        self.gt_answer=[i['conversations'][1]['value'] for i in self.raw_data if i['conversations'][1]['from']=='gpt']

def float_it(score):
    for key in score.keys():
        score[key] = float(score[key])
    return score

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def create_folder(args):
        workpath=os.path.join(args.output_file_path,args.model_path.split('/')[-1])
        if not os.path.exists(workpath):
            os.makedirs(workpath)
        return workpath

def init_json(generation_file):
    with open(generation_file, 'w') as f:
        pass

def write_json(generation_file, user_message, response, ground_truth):
    with open(generation_file, 'a') as f:
        out={'prompt': user_message, 'continuation': response, 'ground_truth': ground_truth}
        f.write(json.dumps(out))
        f.write('\n')


def main(args):
    # Model
    workpath=create_folder(args)
    model = LlavaNextForConditionalGeneration.from_pretrained(args.model_base,torch_dtype=torch.bfloat16,device_map="auto").eval()
    peft_model_id = args.model_path
    try:    
        model.load_adapter(peft_model_id)    
        print(f"Successfully load lora weights to {args.model_base} from {args.model_path}!")

    except:
        print(f"model_base is {args.model_base} while model_path is {args.model_path}! check weather the aim is to evaluate base model or fine-tuned model!")
    processor = LlavaNextProcessor.from_pretrained(args.model_base)

    dataset=gt_dataset(args.dataset_file)
    generation_path=os.path.join(workpath,"results.json")
    init_json(generation_path)

    for (imagedir, text_prompt, gt) in tzip(dataset.image, dataset.user_prompt, dataset.gt_answer):
        if imagedir:
            try:
                image=load_image("../LLaVA/playground/data/"+imagedir)
            except FileNotFoundError:
                image = None
                print("image not loaded! skip to next sample!")
                continue
            if image is not None:
                conversation = [
                {"role": "user", "content": [{"type": "text", "text": text_prompt},{"type": "image"}]},
                ]   
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)         
        else:
            image=None
            conversation = [
            {"role": "user", "content": [{"type": "text", "text": text_prompt}]},
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)  
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,temperature=args.temperature,pad_token_id=processor.tokenizer.eos_token_id)
            if "llama3" in args.model_base:
                outputs = processor.decode(output[0], skip_special_tokens=True).split("assistant\n\n")[-1].strip()
            elif "mistral" in args.model_base:
                outputs = processor.decode(output[0], skip_special_tokens=True).split("[/INST] ")[-1].strip()

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs, "gt_answer": gt }, "\n")
        write_json(generation_path,prompt, outputs, gt)
    data_buffer=[]
    with open(generation_path, 'r') as f:
        for line in f:
            data_buffer.append(json.loads(line))
    with open(generation_path, 'w') as f:
        json.dump(data_buffer, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default="path/to/models--llava-hf--llama3-llava-next-8b-hf/snapshots/hash_code")
    parser.add_argument("--model-path", type=str, default="llava-hf/llama3-llava-next-8b-hf")
    parser.add_argument("--dataset_file", default='path/to/json_file/xstest.json', type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_file_path", type=str, default='../xstest_result',
                        help="Output file_path without ./")

    args = parser.parse_args()
    main(args)