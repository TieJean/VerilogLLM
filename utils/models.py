import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments, AutoConfig
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")

def load_codegen(device: str, type="multi"):
    if type == "mono":
        checkpoint = "Salesforce/codegen-350M-mono"
    elif type == "multi":
        checkpoint = "Salesforce/codegen-350M-multi"
    else:
        raise ValueError(f"Invalid codegen type {type}")
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_gpt2(device: str):
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_llama(device: str, model_type: str = "1B"):
    if model_type == "1B":
        checkpoint = "meta-llama/Llama-3.2-1B"
    elif model_type == "3B":
        checkpoint = "meta-llama/Llama-3.2-3B"
    else:
        raise ValueError(f"Invalid Llama model type: {model_type}. Choose '1B' or '3B'.")
    config = AutoConfig.from_pretrained(checkpoint)
    

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(checkpoint, token=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, token=True)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token
    
    return model, tokenizer