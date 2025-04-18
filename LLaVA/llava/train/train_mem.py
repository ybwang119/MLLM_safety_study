from llava.train.train_without_eos import train
# from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
