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
from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaConfig
import transformers


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
    transformers.models.llava.modeling_llava.LlavaMultiModalProjector = LlavaMultiModalProjectorYiVL
    model_id = args.model_base
    model = AutoModelForVision2Seq.from_pretrained(model_id)
    peft_model_id = args.model_path
    if peft_model_id.split('/')[-1] != "Yi-VL-6B":
        model.load_adapter(peft_model_id)    
    model=model.to(args.device)
    processor = AutoProcessor.from_pretrained(model_id)

    dataset=gt_dataset(args.dataset_file)
    generation_path=os.path.join(workpath,"results.json")
    init_json(generation_path)

    for (imagedir, text_prompt, gt) in tzip(dataset.image, dataset.user_prompt, dataset.gt_answer):
        if imagedir:
            try:
                image=load_image("./mPLUG-Owl/mPLUG-Owl2/playground/data/"+imagedir)
            except FileNotFoundError:
                image = None
                print("image not loaded! skip to next sample!")
                continue
            if image is not None:
                image= [image]
        else:
            inp = text_prompt
            image=None
        messages = [
        { "role": "user", "content": f"{inp}" }
        ]
        text = [processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)]
        with torch.inference_mode():
            inputs = processor(text=text, images=image, return_tensors='pt').to(model.device, torch.float16)
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=args.temperature,
                use_cache=True
                )
            outputs = processor.batch_decode(output, skip_special_tokens=True)[0].split("Assistant: ")[-1].strip()

        if args.debug:
            print("\n", {"prompt": text[0], "outputs": outputs, "gt_answer": gt }, "\n")
        write_json(generation_path, text[0], outputs, gt)
    data_buffer=[]
    with open(generation_path, 'r') as f:
        for line in f:
            data_buffer.append(json.loads(line))
    with open(generation_path, 'w') as f:
        json.dump(data_buffer, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default="BUAADreamer/Yi-VL-6B-hf")
    parser.add_argument("--model-path", type=str, default="./lora/Yi-VL-6B")
    parser.add_argument("--dataset_file", default='../data/dataset/json_file/xstest.json', type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_file_path", type=str, default='./xstest_result',
                        help="Output file_path without ./")

    args = parser.parse_args()
    main(args)