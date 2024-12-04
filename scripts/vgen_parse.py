import json
import os

verilog_blocks = []
current_block = []
block_number = 1
# max_lines = 20000  # Limit to 1000 lines
output_folder = "VGen_verilog_blocks"
os.makedirs(output_folder, exist_ok=True)
outfile = open("output.txt", "w")

# Open the file to read the lines
with open("data/train.csv", "r") as file:
    next(file) 
    for line_number, line in enumerate(file, start=1):
        # if line_number > max_lines:  # Stop reading after 1000 lines
        #     break
        line = line.strip()  
        outfile.write(line + "\n") 

        if "module " in line.lower() and not current_block:
            current_block.append(line)  
        elif "endmodule" in line.lower() and current_block:
            current_block.append(line)  
            verilog_blocks.append({f"block number {block_number}": "\n".join(current_block)}) 
            block_number += 1
            current_block = []  # Reset the current block for the next one
        else: 
            current_block.append(line)

if current_block:
    verilog_blocks.append({f"block number {block_number}": "\n".join(current_block)})


# blocks_per_file = 500
# file_count = 1
# for i in range(0, len(verilog_blocks), blocks_per_file):
#     chunk = verilog_blocks[i:i + blocks_per_file]
#     file_name = os.path.join(output_folder, f"verilog_blocks_part_{file_count}.json")
#     with open(file_name, "w") as json_file:
#         json.dump(chunk, json_file, indent=4)
#     print(f"Saved {len(chunk)} blocks to '{file_name}'.")
#     file_count += 1

single_file_name = os.path.join(output_folder, "verilog_blocks_all.json")
with open(single_file_name, "w") as json_file:
    json.dump(verilog_blocks, json_file, indent=4)

print(f"All Verilog blocks have been saved to the '{output_folder}' folder.")

