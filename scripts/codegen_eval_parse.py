import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.models import *
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./evaluation_data/codegen_solution.jsonl")
    parser.add_argument("--parsed", type=str, default="./evaluation_data/codegen_solution_parsed.jsonl")
    parser.add_argument("--eval_description", type=str, default="./evaluation_data/VerilogDescription_Human.jsonl")

    args = parser.parse_args()
    
    with open(args.output, 'r') as f1, open(args.eval_description, 'r') as f2, open(args.parsed, 'w') as output_file:
        output_lines = [json.loads(line) for line in f1]
        for line in f1:
            data = json.loads(line)
            task = data["task_id"]
            completion = data["completion"]
            
            if "Generated Code:" in completion:
                parts = completion.split("Generated Code:", 1)  
                generated_code = parts[1].lstrip()  # Get the part after "Generated Code:" and strip whitespace
                if ");" in generated_code:
                    parts2 = generated_code.split(");", 1)
                    generated_code = "\n\t" + parts2[1].lstrip()
            else:
                generated_code = ""

            result = {
                "task_id": task,
                "completion": generated_code
            }
           
            output_file.write(json.dumps(result) + "\n")

        for line in f2:
            data2 = json.loads(line)
            task = data2["task_id"]

            # Check if task exists in the parsed JSON
            if any(task == item["task_id"] for item in output_lines):
                pass
            else:
                print(task)  # Task not found





 
    

