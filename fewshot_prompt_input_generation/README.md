# verilogLLM_fewshot

The files in this folder are used to generate the input .json files for inference evaluation of the LLM models for verilog generation. 
The fewshot_csvdb.py script processes the VerilogDescription_Human.jsonl file and adds relevant examples to the prompts using MG-Verilog dataset as the reference dataset. 
Example:

Without fewshot:
input prompt:
"Build a circuit that reverses the byte order of a 32-bit vector."}

With fewshot:
"
Build a circuit that reverses the byte order of a 32-bit vector.Use the following description and code as an example:\nDescription: \n\nThis Verilog module implements a byte-reversal operation for a 64-bit word. It uses a generate block to iterate over the 64 bits and reverse the order of the bytes, storing the result in the output vector q_o. The module has a parameter for the number of bytes in a word and input and output vectors for the word and its reversed form, respectively. The module utilizes nested for-loops to reverse the bytes and assign them to the output vector.\n\n Module header:\n\nmodule dsi_byte_reverse (d_i,q_o);\n [/INST]\n\n

Associated Code:  \n  parameter g_num_bytes = 4; \n  input [g_num_bytes * 8 - 1 : 0] d_i; \n  output [g_num_bytes * 8 - 1 : 0] q_o; \n  generate \n  genvar i, j; \n  for(i=0;i<8;i=i+1) \n  for(j=0;j<g_num_bytes;j=j+1) \n  assign q_o[8 * (g_num_bytes-1-j) + i] = d_i[8 * j + i]; \n  endgenerate \n endmodule
"

Usage: 
pip install -r requirements.txt

python fewshot_csvdb.py
