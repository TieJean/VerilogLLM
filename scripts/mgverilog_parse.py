import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from utils.models import *
from utils.dataloader import *
from peft import LoraConfig, get_peft_model, TaskType

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llama_model, llama_tokenizer = load_llama(device)

    lora_config2 = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Task type
        inference_mode=False,          # Indicates training, not inference
        r=8,                           # Low-rank matrix dimension
        lora_alpha=32,                 # Scaling factor; W' = W = alpha/r * (AB)
        lora_dropout=0.05,              # Dropout for LoRA layers
    )
    llama_model = get_peft_model(llama_model, lora_config2)
    

    mg_dataset = MGDataset("packaged_dataset/simple_description_dataset", llama_tokenizer, max_length=512)

    # Seeing items in dataset
    print(mg_dataset.data[0].keys())
    for i in range(5):
        item = mg_dataset.data[i]  # Access the raw dataset entry
        description = item['description']
        code = item['code']
        print("Description: ", description)
        print("Code: ", code)


        print("-" * 100)