import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

def load_codegen(device: str, type="multi", size="350M"):
    if type == "mono":
        checkpoint = f"Salesforce/codegen-{size}-mono"
    elif type == "multi":
        checkpoint = f"Salesforce/codegen-{size}-multi"
    else:
        raise ValueError(f"Invalid codegen type {type}")
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

def load_gpt2(device: str):
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer