import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from utils.models import *
from utils.dataloader import *
from peft import LoraConfig, get_peft_model, TaskType

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Codegen model training
    # codegen_model, codegen_tokenizer = load_codegen(device, "mono")
    # # TODO need to tune me
    # lora_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,  # Task type
    #     inference_mode=False,          # Indicates training, not inference
    #     r=8,                           # Low-rank matrix dimension
    #     lora_alpha=32,                 # Scaling factor; W' = W = alpha/r * (AB)
    #     lora_dropout=0.05,              # Dropout for LoRA layers
    # )
    # codegen_model = get_peft_model(codegen_model, lora_config)
    # # import pdb; pdb.set_trace()
    # # model.print_trainable_parameters()
    
    # verigen_dataset = VeriGenDataset("data/train.csv", codegen_tokenizer, max_length=1024)
    # train_loader = torch.utils.data.DataLoader(verigen_dataset, batch_size=2)

    
    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     num_train_epochs=15,
    #     per_device_train_batch_size=2,
    #     save_steps=8,
    #     save_total_limit=2,
    #     logging_dir="./logs",
    #     logging_steps=25,
    #     report_to="none",
    # )

    # # Initialize Trainer
    # codegen_trainer = Trainer(
    #     model=codegen_model,
    #     args=training_args,
    #     train_dataset=verigen_dataset,
    # )
    # codegen_trainer.train_loader = train_loader

    # # Fine-tune the model on GPU
    # codegen_trainer.train()
    # codegen_model.save_pretrained("./results/lora_codegen")


    # Llama model training
    llama_model, llama_tokenizer = load_llama(device, "1B")

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
    # print(mg_dataset.data[0].keys())
    # for i in range(5):
    #     item = mg_dataset.data[i]  # Access the raw dataset entry
    #     description = item['description']
    #     code = item['code']
    #     print("Description: ", description)
    #     print("Code: ", code)

    #     # data = mg_dataset[i]
    #     # print("Input IDs:", data['input_ids'])
    #     # print("Attention Mask:", data['attention_mask'])
    #     # print("Labels:", data['labels'])
    #     print("-" * 100)



    train_loader = torch.utils.data.DataLoader(mg_dataset, batch_size=2)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=15,
        per_device_train_batch_size=1,
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=25,
        report_to="none",
    )

    # Initialize Trainer
    llama_trainer = Trainer(
        model=llama_model,
        args=training_args,
        train_dataset=mg_dataset,
    )
    llama_trainer.train_loader = train_loader
    llama_trainer.train()
    llama_model.save_pretrained("./results/lora_llama")


    # # Test model with sample description
    # text = codegen_template(desc="Add two numbers in Verilog.")
    # inputs = codegen_tokenizer(text, return_tensors="pt").to(device)
    # output = codegen_model.generate(
    #     inputs["input_ids"],
    #     max_length=512,        # Maximum length of the output
    #     num_beams=5,           # Beam search for better quality
    #     temperature=0.7,       # Sampling temperature for diversity
    #     top_k=50,              # Limit sampling to top k tokens
    #     top_p=0.95,            # Nucleus sampling for diversity
    #     repetition_penalty=1.2 # Penalize repetitive text
    # )
    # print(codegen_tokenizer.decode(output[0], skip_special_tokens=True))