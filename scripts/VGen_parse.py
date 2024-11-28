import json
import os

verilog_blocks = []
current_block = []
block_number = 1
# max_lines = 20000  # Limit to 1000 lines
output_folder = "data"
os.makedirs(output_folder, exist_ok=True)

# Open the file to read the lines
with open("data/train.csv", "r") as file:
    next(file) 
    for line_number, line in enumerate(file, start=1):
        line = line.strip()  
        if "module " in line.lower() and not current_block:
            current_block.append(line)  
        elif "endmodule" in line.lower() and current_block:
            current_block.append(line)  
            verilog_blocks.append({f"code": "\n".join(current_block)}) 
            block_number += 1
            current_block = []  # Reset the current block for the next one
        else: 
            current_block.append(line)

if current_block:
    verilog_blocks.append({f"code": "\n".join(current_block)})


single_file_name = os.path.join(output_folder, "verilog_blocks_all.json")
with open(single_file_name, "w") as json_file:
    json.dump(verilog_blocks, json_file, indent=4)

print(f"All Verilog blocks have been saved to the '{output_folder}' folder.")