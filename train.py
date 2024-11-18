import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from utils.models import *
from utils.dataloader import *
from peft import LoraConfig, get_peft_model, TaskType

if __name__ == "__main__":
    # TODO move this to argparse
    MAX_LENGTH = 1024
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_codegen(device)
    # TODO need to tune me
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Task type
        inference_mode=False,          # Indicates training, not inference
        r=8,                           # Low-rank matrix dimension
        lora_alpha=32,                 # Scaling factor; W' = W = alpha/r * (AB)
        lora_dropout=0.05,              # Dropout for LoRA layers
    )
    model = get_peft_model(model, lora_config)
    # import pdb; pdb.set_trace()
    # model.print_trainable_parameters()
    
    dataset = VeriGenDataset("data/train.csv", tokenizer, max_length=MAX_LENGTH)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=15,
        per_device_train_batch_size=2,
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=25,
        report_to="none",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train_loader = train_loader

    # Fine-tune the model on GPU
    trainer.train()
    model.save_pretrained("./results/verilogLLM")

    text = codegen_template(desc="Add two numbers in Verilog")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output = model.generate(
        inputs["input_ids"],
        max_length=MAX_LENGTH,        # Maximum length of the output
        num_beams=5,           # Beam search for better quality
        temperature=0.7,       # Sampling temperature for diversity
        top_k=50,              # Limit sampling to top k tokens
        do_sample=True,
        top_p=0.95,            # Nucleus sampling for diversity
        repetition_penalty=1.2, # Penalize repetitive text
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))