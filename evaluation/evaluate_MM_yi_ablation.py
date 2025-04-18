import argparse
import os
from tqdm import tqdm
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

    dataset,subfield_name,dataset_size=create_MM_safetybench_subdataset(args)
    name=args.input_mode
    generation_path=os.path.join(workpath,subfield_name+"_"+name+".json")
    init_json(generation_path)
    for i in tqdm(dataset):
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
            image= [image]
        inp = "<image>" + '\n' + text_prompt
        messages = [
        { "role": "user", "content": f"{inp}" }
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

        if args.debug:
            print("\n", {"prompt": "", "outputs": outputs}, "\n")
        # write_json(generation_path, prompt, outputs, detoxify_scorer, perspective_scorer)
        write_json(generation_path, text[0], outputs)
    data_buffer=[]
    with open(generation_path, 'r') as f:
        for line in f:
            data_buffer.append(json.loads(line))
    with open(generation_path, 'w') as f:
        json.dump(data_buffer, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default="BUAADreamer/Yi-VL-6B-hf")
    parser.add_argument("--model-path", type=str, default="./lora/Yi-VL-6B-my1")
    parser.add_argument("--dataset_file", default='../data/dataset/MM-SafetyBench/data', type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_file_path", type=str, default='MM_bench_result',
                        help="Output file_path without ./")
    parser.add_argument("--index",type=int, default=7, help="specify the index of subfield.")
    parser.add_argument("--input-mode", default="mix",choices=['sd','ocr','mix','pure_text','text2img','img2text','trial','matched_ocr'], help="specify the input mode.")               
    parser.add_argument("--insert-mode", default=False,choices=['prepend','append','insert','random_insert',False], help="specify the image_imbedding_editing_mode. Only works for img2text mode")               

    args = parser.parse_args()
    main(args)