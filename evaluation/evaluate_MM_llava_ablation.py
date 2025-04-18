import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES']="4"
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


class MM_sample():
    def __init__(self, OCR, SD, mix, text, text_SD, pure_text, matched_ocr):
        self.OCR=OCR
        self.SD=SD
        self.mix=mix
        self.text=text
        self.text_SD=text_SD
        self.matched_ocr=matched_ocr
        self.pure_text=pure_text
        self.modified_text=self.text_SD.replace("a phrase of","a request about")
    def get_OCR(self):
        return self.OCR, self.text
    def get_SD(self):
        return self.SD, self.text_SD
    def get_mix(self):
        return self.mix,self.text
    def get_text(self):
        return None, self.pure_text
    def get_text2img(self):
        return self.matched_ocr, None
    def get_img2text(self):
        return self.pure_text, self.modified_text
    def get_matched_ocr(self):
        return self.matched_ocr, self.modified_text
def create_MM_safetybench_subdataset(args):
    dataset = []
    query_path=os.path.join(args.dataset_file,"processed_questions")
    query_subfield=os.listdir(query_path)
    query_subfield=sorted(query_subfield,key=lambda x:int(x[:2]))
    query_file = open(os.path.join(query_path,query_subfield[args.index])).read()
    queries=json.loads(query_file)
    SD_path=os.path.join(args.dataset_file,"imgs",query_subfield[args.index].split(".")[0],"SD")
    OCR_path=os.path.join(args.dataset_file,"imgs",query_subfield[args.index].split(".")[0],"TYPO")
    mix_path=os.path.join(args.dataset_file,"imgs",query_subfield[args.index].split(".")[0],"SD_TYPO")
    matched_ocr_path=os.path.join(args.dataset_file,"imgs",query_subfield[args.index].split(".")[0],"matched_ocr")
    for ind in queries.keys():
        text=queries[ind]["Rephrased Question"]
        text_SD=queries[ind]["Rephrased Question(SD)"]
        pure_text=queries[ind]["Changed Question"]
        OCR=Image.open(os.path.join(OCR_path,f"{ind}.jpg")).convert('RGB')
        SD=Image.open(os.path.join(SD_path,f"{ind}.jpg")).convert('RGB')
        mix=Image.open(os.path.join(mix_path,f"{ind}.jpg")).convert('RGB')
        matched_ocr=Image.open(os.path.join(matched_ocr_path,f"{ind}.jpg")).convert('RGB')
        dataset.append(MM_sample(OCR,SD,mix,text,text_SD,pure_text,matched_ocr))
    return dataset,query_subfield[args.index].split(".")[0],len(queries.keys())

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
        os.makedirs(workpath,exist_ok=True)
        return workpath

def init_json(generation_file):
    with open(generation_file, 'w') as f:
        pass

def write_json(generation_file, user_message, response):
    with open(generation_file, 'a') as f:
                out={'prompt': user_message, 'continuation': response}
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
import torch
import random

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


    dataset,subfield_name,dataset_size=create_MM_safetybench_subdataset(args)
    name=args.input_mode
    generation_path=os.path.join(workpath,subfield_name+"_"+name+".json")
    init_json(generation_path)
    for i in dataset:
        conv = conv_templates[args.conv_mode].copy()
        if args.input_mode=='img2text': # wrap malicious input with image sturcture. Use multimodal prompt
            image_like_text,text_prompt = i.get_img2text()
            if model.config.mm_use_im_start_end:
                inp = text_prompt + '\n'+ DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            else:
                inp = text_prompt + '\n' +DEFAULT_IMAGE_TOKEN 
            # if model.config.mm_use_im_start_end:
            #     inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text_prompt
            # else:
            #     inp = DEFAULT_IMAGE_TOKEN + '\n' + text_prompt
        elif args.input_mode=='pure_text':# directly ask the model to generate text with malicious prompt
            image,text_prompt = i.get_text()
            inp = text_prompt 
            image_tensor=None
            image_size=None 
        elif args.input_mode=='trial':# directly ask the model to generate text with malicious prompt
            image_like_text,text_prompt = i.get_img2text()
            image=Image.open('white.jpg').convert('RGB')
            image_size = image.size
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            # inp = DEFAULT_IMAGE_TOKEN +image_like_text+ '\n'  +text_prompt 
            inp = text_prompt+ '\n'+DEFAULT_IMAGE_TOKEN +image_like_text
        else:
            if args.input_mode=='ocr':
                image,text_prompt = i.get_OCR()
            elif args.input_mode=='sd':
                image,text_prompt = i.get_SD()
            elif args.input_mode=='mix':
                image,text_prompt = i.get_mix()
            elif args.input_mode=='text2img':##simply use image without text prompt
                image,text_prompt = i.get_text2img()
            elif args.input_mode=='matched_ocr':
                image,text_prompt = i.get_matched_ocr()
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
        # image = None
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            if name=='img2text':
                image_like_token=tokenizer(image_like_text,return_tensors='pt').input_ids[:,1:] #delete the <s> token
                image_like_features=model.get_model().embed_tokens(image_like_token)
                if args.insert_mode:
                    black_feature=torch.load('white_feature.pt').to(model.device)
                    image_like_features=merge_tensors(image_like_features,black_feature,args.insert_mode)
                output_ids,_,_,_,new_input_embeds,_=generate_from_word_tensor(args, model, image_like_features, input_ids,streamer)
            else:
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
            print("\n", {"prompt": prompt, "outputs": outputs[4:-4]}, "\n")
        # write_json(generation_path, prompt, outputs, detoxify_scorer, perspective_scorer)
        write_json(generation_path, prompt, outputs[4:-4])
    data_buffer=[]
    with open(generation_path, 'r') as f:
        for line in f:
            data_buffer.append(json.loads(line))
    with open(generation_path, 'w') as f:
        json.dump(data_buffer, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lora/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset_file", default='../data/dataset/MM-SafetyBench/data', type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_file_path", type=str, default='MM_bench_result',
                        help="Output file_path without ./")
    parser.add_argument("--index",type=int, default=7, help="specify the index of subfield.")
    parser.add_argument("--input-mode", default="mix",choices=['sd','ocr','mix','pure_text','text2img','img2text','trial','matched_ocr'], help="specify the input mode.")               
    parser.add_argument("--insert-mode", default=False,choices=['prepend','append','insert','random_insert',False], help="specify the image_imbedding_editing_mode. Only works for img2text mode")               

    args = parser.parse_args()
    main(args)