import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.models import *
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./evaluation_data/llama_solution.jsonl")
    parser.add_argument("--parsed", type=str, default="./evaluation_data/llama_solution_parsed.jsonl")

    args = parser.parse_args()
    with open(args.output, 'r') as f1, open(args.parsed, 'w') as output_file:
        for line in f1:
            data = json.loads(line)
            task = data["task_id"]
            completion = data["completion"]
            
            if "Response:" in completion:
                parts = completion.split("Response:", 1)  
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
