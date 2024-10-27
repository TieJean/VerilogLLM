import torch
from transformers import Trainer, TrainingArguments
from utils.models import *
from utils.dataloader import *
from datasets import Dataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_gpt2(device)
    dataset = VeriGenDataset("data/train.csv", tokenizer, max_length=1024)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train_loader = train_loader

    # Fine-tune the model on GPU
    # TODO: may want to do LoRA
    trainer.train()

    text = f"Prompt: Add two numbers in Verilog."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    completion = model.generate(**inputs)
    print(tokenizer.decode(completion[0]))