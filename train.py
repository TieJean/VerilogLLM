import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from utils.models import *
from utils.dataloader import *
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datetime import datetime

class LossLoggingCallback(TrainerCallback):
    def __init__(self, outpath):
        self.outpath = outpath
        self.loss_log = []

    def on_step_end(self, args, state, control, **kwargs):
        # Log loss at each step
        if state.log_history and "loss" in state.log_history[-1]:
            loss = state.log_history[-1]["loss"]
            step = state.global_step
            self.loss_log.append({"step": step, "loss": loss})
    
    def on_train_end(self, args, state, control, **kwargs):
        # Save the loss log to a file at the end of training
        with open(self.outpath, "w") as f:
            import json
            json.dump(self.loss_log, f)
        print(f"Training loss log saved to {self.outpath}")

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="codegen")
    parser.add_argument("--model-size", type=str, default="350M")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    MAX_LENGTH = args.max_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "codegen":
        base_model, tokenizer = load_codegen(device, size=args.model_size)
    elif args.model_type == "llama":
        base_model, tokenizer = load_llamaInstruct(device)
    else:
        print(f"Invalid model type {args.model_type}")
        exit(1)
         
    # TODO need to tune me
    if args.checkpoint is None:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Task type
            inference_mode=False,          # Indicates training, not inference
            r=8,                           # Low-rank matrix dimension
            lora_alpha=32,                 # Scaling factor; W' = W = alpha/r * (AB)
            lora_dropout=0.05,              # Dropout for LoRA layers
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
    else:
        print(f"Loading checkpoint from {args.checkpoint}...")
        model = PeftModel.from_pretrained(base_model, "./results/verilogLLM-codegen-350M/")
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        model.print_trainable_parameters()
    # model.print_trainable_parameters()
    
    # datapaths = ["data/verilog_blocks_all.json"]
    datapaths = [
         "data/verilog_blocks_all.json",
        "data/packaged_dataset/detailed_description_dataset/",
        "data/packaged_dataset/simple_description_dataset/",
        "data/packaged_dataset/merged_dataset/",
        "data/packaged_dataset/llm2_block_summary_to_pure_code_one_shot_dataset/",
        "data/packaged_dataset/vanilla_baseline/"
    ]
    dataset = VeriGenDataset(datapaths, tokenizer, max_length=MAX_LENGTH)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    savename = f"verilogLLM-{args.model_type}-{args.model_size}-{datetime.today().strftime('%Y%m%d')}"
    per_device_train_batch_size = args.batch_size // torch.cuda.device_count()
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[LossLoggingCallback(os.path.join("./logs", f"{savename}.json"))],
    )
    trainer.train_loader = train_loader

    # Fine-tune the model on GPU
    trainer.train()
    savepath = f"./results/verilogLLM-{args.model_type}-{args.model_size}"
    if args.checkpoint is not None:
        savepath += f"-{datetime.today().strftime('%Y%m%d')}"
    model.save_pretrained(savepath)

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