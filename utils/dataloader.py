import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os

class VeriGenDataset(Dataset):
    def __init__(self, datapath: str, tokenizer: AutoTokenizer, max_length: int):
        if not os.path.exists(datapath):
            raise FileNotFoundError(f"The file '{datapath}' does not exist.")
        # TODO load post-processed data
        # dummy data
        self.data = [
            {"prompt": "Add two numbers in Verilog.", "response": "module add(input [3:0] a, b, output [3:0] sum);\nassign sum = a + b;\nendmodule"},
            {"prompt": "4-bit counter in Verilog.", "response": "module counter(output reg [3:0] count, input clk);\nalways @(posedge clk) count <= count + 1;\nendmodule"},
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # TODO need to have a text template function
        text = f"Prompt: {item['prompt']} {self.tokenizer.eos_token} Verilog Code: {item['response']}"

        # Tokenize text
        encoding = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True
        )

        # Use input_ids as labels for causal language modeling
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # The labels should be the same as input_ids for causal LM tasks
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }