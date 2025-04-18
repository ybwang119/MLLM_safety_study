import argparse
import os
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import nn
import json
import requests
from PIL import Image
from io import BytesIO
import transformers
import numpy as np
from tqdm import tqdm
    
def init_json(generation_file):
    with open(generation_file, 'w') as f:
        pass

def write_json(generation_file, user_message, response, ground_truth,score):
    with open(generation_file, 'a') as f:
        out={'prompt': user_message, 'continuation': response, 'ground_truth': ground_truth, 'score': score}
        f.write(json.dumps(out))
        f.write('\n')

class VizWizEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image']
        question = data['question']
        answers = data['answers']
        answers = '_'.join([answer['answer'] for answer in answers])
        image_path = os.path.join(self.root_path, img_id)
        image = Image.open(image_path).convert('RGB')
        question = f"{question}\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase."
        return image, question, answers

def main(args):
    vizwiz = json.load(open(args.eval_file_path, 'r'))
    model = LlavaNextForConditionalGeneration.from_pretrained(args.model_base,torch_dtype=torch.bfloat16,device_map="auto").eval()
    peft_model_id = args.model_path
    try:    
        model.load_adapter(peft_model_id)    
        print(f"Successfully load lora weights to {args.model_base} from {args.model_path}!")

    except:
        print(f"model_base is {args.model_base} while model_path is {args.model_path}! check weather the aim is to evaluate base model or fine-tuned model!")
    processor = LlavaNextProcessor.from_pretrained(args.model_base)

    output_dir = f"{args.output_file_path}/{peft_model_id.split('/')[-1]}"
    os.makedirs(output_dir, exist_ok=True)
    generation_path=os.path.join(output_dir, "generation.json")
    output_file_path = os.path.join(output_dir, "summary.txt")
    data = VizWizEvalData(vizwiz, None, args.dataset_file)
    total_acc = []
    init_json(generation_path)
    for index in tqdm(range(len(data))):
        image, text_prompt, gt_answers = data[index]
        conversation = [
        {"role": "user", "content": [{"type": "text", "text": text_prompt},{"type": "image"}]},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,temperature=args.temperature,pad_token_id=processor.tokenizer.eos_token_id)

            outputs = processor.decode(output[0], skip_special_tokens=True).split("[/INST] ")[-1].strip()
        for answer, gt_answer in zip([outputs], [gt_answers]):
            count=0
            gt_answer = gt_answer.split('_')
            for gt in gt_answer:
                if gt.lower() == answer.lower():
                    count += 1
            acc = min(count/3.0, 1.0)
            total_acc.append(acc)
        write_json(generation_path,prompt,outputs,gt_answers,acc)
    data_buffer=[]
    with open(generation_path, 'r') as f:
        for line in f:
            data_buffer.append(json.loads(line))
    with open(generation_path, 'w') as f:
        json.dump(data_buffer, f, indent=4)
    with open(output_file_path, "w") as output_file:
        output_file.write(f"total sample size: {len(total_acc)}, vizwiz Acc: {np.average(total_acc)* 100.0}\n")
    print('vizwiz Acc: ', np.average(total_acc)* 100.0, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default="models--llava-hf--llama3-llava-next-8b-hf/snapshots/xxxxxxxxxx")
    parser.add_argument("--model-path", type=str, default="llava-hf/llama3-llava-next-8b-hf")
    parser.add_argument("--dataset_file", default='../../data/dataset/vizwiz/vizwiz_validation', type=str)
    parser.add_argument('--eval_file_path', type=str, default='../../data/dataset/vizwiz/val.json')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output_file_path", type=str, default='../vizwiz_result',
                        help="Output file_path without ./")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()    
    main(args)