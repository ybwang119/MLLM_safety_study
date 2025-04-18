# code for evaluating the VizWiz dataset, borrowed from https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/datasets/datasets/vqa_datasets.py#L46.
# The code is modified to evaluate the VizWiz dataset using the LlaVA model.
import argparse
import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from PIL import Image
import json
import numpy as np
from transformers import TextStreamer
from torch.utils.data import DataLoader
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
        # image = self.vis_processor(image)
        # question = f"[vqa] The question is '{question}' Based on the image, answer the question with a single word or phrase. and reply 'unanswerable' when the provided information is insufficient"
        #llava official prompt
        question = f"{question}\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase."
        return image, question, answers

    

def main(args):
    
    disable_torch_init()
    vizwiz = json.load(open(args.eval_file_path, 'r'))
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    output_dir = f"{args.output_file_path}/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    generation_path=os.path.join(output_dir, "generation.json")
    output_file_path = os.path.join(output_dir, "summary.txt")
    data = VizWizEvalData(vizwiz, image_processor, args.dataset_file)
    conv = conv_templates[args.conv_mode].copy()
    total_acc = []
    init_json(generation_path)
    for index in range(len(data)):
        conv = conv_templates[args.conv_mode].copy()
        image, text_prompt, gt_answers = data[index]
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                processed_inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text_prompt
            else:
                processed_inp = DEFAULT_IMAGE_TOKEN + '\n' + text_prompt
            image = None
        
        conv.append_message(conv.roles[0], processed_inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        conv.messages[-1][-1] = outputs
        
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
    parser.add_argument("--model-path", type=str, default="ckpts/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset_file", default='../data/dataset/vizwiz/vizwiz_validation', type=str)
    parser.add_argument('--eval_file_path', type=str, default='../data/dataset/vizwiz/val.json')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_file_path", type=str, default='./vizwiz_result',
                        help="Output file_path without ./")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    # api_key = open(".perspective_api_key").read().strip()
    
    main(args)
