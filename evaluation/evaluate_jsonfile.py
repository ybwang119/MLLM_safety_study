import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES']="7"
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,IGNORE_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
import json
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer, LlamaForCausalLM
import random

class gt_dataset:
    def __init__(self, json_file):
        self.filedir = json_file
        DEFAULT_IMAGE_TOKEN = None
        with open(json_file, 'r') as f:
            self.raw_data = json.load(f)
        self.image=[i['image'] for i in self.raw_data]
        for i in ["<image>",'<image\n>']:
            if i in self.raw_data[0]['conversations'][0]['value']:
                DEFAULT_IMAGE_TOKEN=i
                break
        if DEFAULT_IMAGE_TOKEN is not None:
            self.user_prompt=[i['conversations'][0]['value'].split(DEFAULT_IMAGE_TOKEN)[-1] for i in self.raw_data if i['conversations'][0]['from']=='human']
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
def generate_from_word_tensor(args, model:LlavaLlamaForCausalLM, image_like_tensor, input_ids,streamer):
    position_ids=None
    attention_mask = None
    past_key_values = None
    labels = None
    # image_features=model.get_model().embed_tokens(torch.tensor(tokenizer(image_like_text).input_ids, dtype=torch.long, device=model.device))
    image_features=image_like_tensor
    print(f'input feature shape: {image_features.shape}')

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(model.config, 'tune_mm_mlp_adapter', False) and getattr(model.config, 'mm_use_im_start_end', False):
        raise NotImplementedError

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- FIXME
    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = model.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = model.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []
# 找到image对应token切分，然后三块拼在一起计算embedding（可能比较快），然后再切回去
        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

        cur_new_input_embeds = [x.to(model.device) for x in cur_new_input_embeds]

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(model.config, 'tokenizer_model_max_length', None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(model.config, 'tokenizer_padding_side', 'right') == "left":
            new_input_embeds_padded.append(torch.cat((
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                cur_new_embed
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None
    output_ids=super(LlamaForCausalLM, model).generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=new_input_embeds,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            use_cache=True
        )
    return output_ids, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
# MM_results
    # model_name
        #01_SD.json
        #02_mix

def merge_tensors(source: torch.Tensor, ingredient: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == 'prepend':
        # 插在最前
        return torch.cat([ingredient, source], dim=1)
    elif mode == 'append':
        # 插在最后
        return torch.cat([source, ingredient], dim=1)
    elif mode == 'insert':
        # 插在任意位置
        position = random.randint(0, source.size(1))
        return torch.cat([source[:,:position], ingredient[:,], source[:,position:]], dim=1)
    elif mode == 'random_insert':
        # 将tensor中每一个子向量插入随机的位置
        for i in range(ingredient.size(1)):
            position = random.randint(0, source.size(1))
            source = torch.cat([source[:,:position], ingredient[:,i:i+1], source[:,position:]], dim=1)
        return source
    else:
        raise ValueError("Invalid mode. Choose from 'prepend', 'append', 'insert', 'random_insert'.")

def main(args):
    # Model
    workpath=create_folder(args)
    disable_torch_init()
    
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)
    # perspective_scorer = PerspectiveApiScorer(api_key)
    # detoxify_scorer = DetoxifyScorer(device=model.device)
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


    dataset=gt_dataset(args.dataset_file)
    generation_path=os.path.join(workpath,"results.json")
    init_json(generation_path)

    for (imagedir, text_prompt, gt) in zip(dataset.image, dataset.user_prompt, dataset.gt_answer):
        conv = conv_templates[args.conv_mode].copy()
        if imagedir:
            try:
                image=load_image("./LLaVA/playground/data/"+imagedir)
            except FileNotFoundError:
                image = None
                print("image not loaded! skip to next sample!")
                continue
            if image is not None:
                image_size = image.size
                image_tensor = process_images([image], image_processor, model.config)
                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text_prompt
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + text_prompt if text_prompt is not None else DEFAULT_IMAGE_TOKEN+ '\n'
                image = None
        else:
            inp = text_prompt
            image_tensor = None
            image_size = None
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size] if image_size is not None else None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs[4:-4], "gt_answer": gt }, "\n")
        write_json(generation_path, prompt, outputs[4:-4], gt)
    data_buffer=[]
    with open(generation_path, 'r') as f:
        for line in f:
            data_buffer.append(json.loads(line))
    with open(generation_path, 'w') as f:
        json.dump(data_buffer, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="ckpts/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset_file", default='../data/dataset/json_file/xstest.json', type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_file_path", type=str, default='xstest_result',
                        help="Output file_path without ./")
    args = parser.parse_args()
    main(args)