# Do we really need curated malicious data for safety alignment in multi-modal large language models?

This repo contains the official code for CVPR 2025 paper\
[Do we really need curated malicious data for safety alignment in multi-modal large language models?](https://arxiv.org/abs/2504.10000)
![Data construction wrokflow](./assets/summary_cm.png)
Multi-modal large language models (MLLMs) have made significant progress, yet their safety alignment remains limited. Typically, current open-source MLLMs rely on the alignment inherited from their language module to avoid harmful generations. However, the lack of safety measures specifically designed for multi-modal inputs creates an alignment gap, leaving MLLMs vulnerable to vision-domain attacks such as typographic manipulation. Current methods utilize a carefully designed safety dataset to enhance model defense capability, while the specific knowledge or patterns acquired from the high-quality dataset remain unclear.
Through comparison experiments, we find that the alignment gap primarily arises from data distribution biases, while image content, response quality, or the contrastive behavior of the dataset makes little contribution to boosting multi-modal safety. To further investigate this and identify the key factors in improving MLLM safety, we propose finetuning MLLMs on a small set of benign instruct-following data with responses replaced by simple, clear rejection sentences. Experiments show that, without the need for labor-intensive collection of high-quality malicious data, model safety can still be significantly improved, as long as a specific fraction of rejection data exists in the finetuning set, indicating the security alignment is not lost but rather obscured during multi-modal pretraining or instruction finetuning. Simply correcting the underlying data bias could narrow the safety gap in the vision domain.
## Requirements

### Installation

Create a conda environment and install dependencies:

```shell
git clone https://github.com/ybwang119/MLLM_safety_study.git
cd MLLM_safety_study
```
**Here, we recommend users to follow the official codespace of [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA) to create the model finetuning environment**, including the competible version of torch, huggingface, flash_attn, bitsandbytes, etc. For LLaVA-NeXT training, please refer to the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
Therefore, we do not present explicit `requirement.txt` for environment creating.
### Datasets and Models

It is okay to download the related data following the instrution in [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA). In addition, we also present a dataset bundle in huggingface, including LLaVA-Med, vizwiz, vg, scienceqa, coco, ocr-vqa, textvqa, MM-SafetyBench, XSTest.

For finetuning dataset arrangement, we utilize `mix_rejection_data.py` to mix LLaVA-Med data with the original instruct tuning data, in which we change the replies. 

```shell
cd utils
python mix_rejection_data.py --file1 path_to_llavamed_json --file2 path_to_mix665k_data --output output_folder --sample_size1 2000 --sample_size2 5000 --change_reply --suffix "your_datafile_identifier" --randomchoose
```

For Llama-factory, remember to first add the related dataset config in `LLaMA-Factory/data/dataset_info.json`. For example:

```json
{
  ...
  "2000_vlguard_train_llava_format_5000_llava_v1_5_mix665k_one_turn_tr": {
        "file_name": "2000_vlguard_train_llava_format_5000_llava_v1_5_mix665k_one_turn_tr.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "image"
        }
    },
}
```

We provide the `utils/add_list.ipynb` to make the json file compatible with llama-factory finetuning framework.
## File Replacement

The key difference between the proposed method and official next-token-prediction pipeline is that we only consider a few rejection tokens at the beginning. 

1. Please add the `train_without_eos.py` folder in LLaVA official code under `llava/train`
2. modify the `llava/train/train_mem.py`
```python
from llava.train.train_without_eos import train
# from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
```
Main modifications of `train_without_eos.py` lie in [Line 494](https://github.com/ybwang119/MLLM_safety_study/blob/b312b374f86fd4ee31200db995a7e16ed349ef8e/LLaVA/llava/train/train_without_eos.py#L494) 

```python
def preprocess_v1(...):
  ...
  for conversation, target in zip(conversations, targets):
    ...
    if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

      if "I'm sorry, but I cannot assist with your request" in conversation:   
                target[cur_len-1] = IGNORE_INDEX


      return dict(
          input_ids=input_ids,
          labels=targets,
      )

```
Similarly, in LLaMA-Factory we modify `LLaMA-Factory/src/llamafactory/data/processors/supervised.py`, main modifications are in [Line 83](https://github.com/ybwang119/MLLM_safety_study/blob/b312b374f86fd4ee31200db995a7e16ed349ef8e/LLaMA-Factory/src/llamafactory/data/processors/supervised.py#L83).
## Training and Evaluation
  It is okay to directly use official code for training. Also, we modify the training script based on the official code, where you could finetune models with different dataset json files and automatically find the available cuda devices. For default setting, we use four Nvidia GPUs with each memory larger than 45GB for LoRA finetuning.
  ```shell
  cd LLaVA
  ./scripts/v1_5/finetune_task_lora.sh
  ```
## Citation
```latex
@inproceedings{wang2025do,
  title={Do We Really Need Curated Malicious Data for Safety Alignment in Multi-modal Large Language Models?},
  author={Wang, Yanbo and Guan, Jiyang and Liang, Jian and He, Ran},
  booktitle={2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## Contact

If you have any question, feel free to contact 📫ybwang119@gmail.com.








