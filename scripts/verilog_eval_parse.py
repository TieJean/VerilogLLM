import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.models import *
import json
import argparse

if __name__ == "__main__":
    with open('./evaluation_data/llama_solution.jsonl', 'r') as f1, open('./evaluation_data/llama_solution_parsed.jsonl', 'w') as output_file:
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

            # unwanted_strings = [
            #     "[/ASSIGN]", "[/INPUT]", "[/OUTPUT]", "[/PORT]",
            #     "[/VERilog]", "[/VERILATOR TASK]", "[/TASK]",
            #     "[/Karnaugh_map]", "Response:", "[/VERilogCode]", "[/MARK]", "[/Karnaugh_Map]", "[/B]",
            #     "[/A]", "[/KARNAUGH MAP]", "[/ONEHOT]", "[/GENERIC MSG]"
            # ]
            # for unwanted in unwanted_strings:
            #     generated_code = generated_code.replace(unwanted, "").strip()

            result = {
                "task_id": task,
                "completion": generated_code
            }
           
            output_file.write(json.dumps(result) + "\n")
