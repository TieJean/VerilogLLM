from utils.models import *
from utils.dataloader import *
from peft import PeftModel
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./results/verilogLLM-llama-1B/")
    parser.add_argument("--eval_description", type=str, default="./evaluation_data/VerilogDescription_Human.jsonl")
    parser.add_argument("--eval_tests", type=str, default="./evaluation_data/VerilogEval_Human.jsonl")
    parser.add_argument("--output_file", type=str, default="./evaluation_data/llama_solution.jsonl")  
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, tokenizer = load_llama(device)
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    count = 0
    with open(args.eval_description, 'r') as f1, open(args.eval_tests, 'r') as f2, open(args.output_file, 'w') as output_file:
        
        f2_data = {}
        for line in f2:
            data2 = json.loads(line)
            f2_data[data2["task_id"]] = data2  # Store the whole object keyed by task_id

        for line in f1:
            # if count == 5:
            #     break
            # count += 1
            data = json.loads(line)
            task = data["task_id"]
            task_data_from_f2 = f2_data[task]
            io_ports = task_data_from_f2['prompt']
            description = data["detail_description"]

            prompt = llama_template(desc=description, header=io_ports)

            if(len(prompt)>2048):
                continue
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate code
            output = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=2048,        # Maximum length of the output
                num_beams=6,           # Beam search for better quality
                temperature=0.8,       # Sampling temperature for diversity
                top_k=60,              # Limit sampling to top k tokens
                do_sample=True,
                top_p=0.95,            # Nucleus sampling for diversity
                repetition_penalty=1.9, # Penalize repetitive text
            )

            # Decode and print the generated code
            completion = tokenizer.decode(output[0], skip_special_tokens=True)
            
            result = {
                "task_id": task,
                "completion": completion
            }

            output_file.write(json.dumps(result) + "\n")