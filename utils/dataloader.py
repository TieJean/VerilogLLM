import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
import json

def codegen_template(desc: str, code: str = ""):
    input_text = f"Task: Write Verilog program for the given description.\nDescription: {desc}.\nGenerated Code:\n"
    target_text = code
    return input_text + target_text

def code_template(code: str):
    text = f"Verilog Code: {code}"
    return text

class VeriGenDataset(Dataset):
    def __init__(self, datapath: str, tokenizer: AutoTokenizer, max_length: int):
        if not os.path.exists(datapath):
            raise FileNotFoundError(f"The file '{datapath}' does not exist.")
        self.data = [
            {"prompt": "Add two numbers in Verilog.", "response": "module add(input [3:0] a, b, output [3:0] sum);\nassign sum = a + b;\nendmodule"},
            {"prompt": "4-bit counter in Verilog.", "response": "module counter(output reg [3:0] count, input clk);\nalways @(posedge clk) count <= count + 1;\nendmodule"},
            {"prompt": "2-to-1 multiplexer in Verilog.", "response": "module mux(input a, b, sel, output y);\nassign y = sel ? b : a;\nendmodule"},
            {"prompt": "AND gate in Verilog.", "response": "module and_gate(input a, b, output y);\nassign y = a & b;\nendmodule"},
            {"prompt": "D flip-flop in Verilog.", "response": "module dff(output reg q, input d, clk);\nalways @(posedge clk) q <= d;\nendmodule"},
            {"prompt": "8-bit register in Verilog.", "response": "module reg8(output reg [7:0] q, input [7:0] d, clk);\nalways @(posedge clk) q <= d;\nendmodule"},
            {"prompt": "Full adder in Verilog.", "response": "module full_adder(input a, b, cin, output sum, cout);\nassign {cout, sum} = a + b + cin;\nendmodule"},
            {"prompt": "Half adder in Verilog.", "response": "module half_adder(input a, b, output sum, cout);\nassign sum = a ^ b;\nassign cout = a & b;\nendmodule"},
            {"prompt": "4-to-1 multiplexer in Verilog.", "response": "module mux4(input [3:0] d, input [1:0] sel, output y);\nassign y = d[sel];\nendmodule"},
            {"prompt": "Binary to gray code converter in Verilog.", "response": "module bin_to_gray(input [3:0] bin, output [3:0] gray);\nassign gray = bin ^ (bin >> 1);\nendmodule"},
            {"prompt": "8-bit shift register in Verilog.", "response": "module shift_reg(output reg [7:0] q, input d, clk);\nalways @(posedge clk) q <= {q[6:0], d};\nendmodule"},
            {"prompt": "Decoder for 3-to-8 in Verilog.", "response": "module decoder3to8(input [2:0] in, output [7:0] out);\nassign out = 1 << in;\nendmodule"},
            {"prompt": "Simple state machine with two states in Verilog.", "response": "module fsm(output reg state, input clk, reset);\nalways @(posedge clk or posedge reset) if (reset) state <= 0; else state <= ~state;\nendmodule"},
            {"prompt": "4-bit comparator in Verilog.", "response": "module comparator(input [3:0] a, b, output lt, gt, eq);\nassign lt = a < b;\nassign gt = a > b;\nassign eq = a == b;\nendmodule"},
            {"prompt": "Odd parity generator in Verilog.", "response": "module odd_parity(input [3:0] in, output parity);\nassign parity = ^in;\nendmodule"},
            {"prompt": "Clock divider by 2 in Verilog.", "response": "module clock_div2(output reg clk_out, input clk);\nalways @(posedge clk) clk_out <= ~clk_out;\nendmodule"},
            {"prompt": "Ripple carry adder for 4 bits in Verilog.", "response": "module ripple_carry_adder(input [3:0] a, b, output [4:0] sum);\nassign sum = a + b;\nendmodule"},
            {"prompt": "Simple ALU with addition and AND operation in Verilog.", "response": "module alu(input [3:0] a, b, input sel, output [3:0] result);\nassign result = sel ? (a + b) : (a & b);\nendmodule"},
            {"prompt": "7-segment display decoder for BCD in Verilog.", "response": "module seven_seg(input [3:0] bcd, output reg [6:0] seg);\nalways @* case(bcd)\n4'b0000: seg = 7'b1000000; // 0\n4'b0001: seg = 7'b1111001; // 1\n// Other cases here\nendcase\nendmodule"},
            {"prompt": "Synchronous 2-bit counter in Verilog.", "response": "module sync_counter(output reg [1:0] count, input clk, reset);\nalways @(posedge clk or posedge reset) if (reset) count <= 0; else count <= count + 1;\nendmodule"}
        ]
        file = open(datapath, "r")
        data = json.load(file)
        self.data.append(data)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if "prompt" in item.keys():
            text = codegen_template(desc=item['prompt'], code=item['response'])
        else:
            text = code_template(item["code"])

        # Tokenize the inputs and targets
        embeddings = self.tokenizer(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        # Use input_ids as labels for causal language modeling
        input_ids = embeddings["input_ids"].squeeze()
        attention_mask = embeddings["attention_mask"].squeeze()
        labels = embeddings["input_ids"].squeeze().clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }