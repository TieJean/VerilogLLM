Tested with Python 3.8.10 and pip 20.0.2

## Install Dependency
```
pip install -r requirements.txt
```

## Example Workflow for Evaluation
```
git clone https://github.com/TieJean/VerilogLLM.git
cd VerilogLLM

<init your python/conda virtual environment>
pip install -r requirements.txt

python verilog_eval_llama.py 

python scripts/verilog_eval_parse.py 

cd verilog-eval/verilog_eval
python evaluate_functional_correctness.py ../../evaluation_data/llama_solution_parsed.jsonl --problem_file=../../evaluation_data/VerilogEval_Human.jsonl

cd ../../evaluation_data
nano llama_solution_parsed.json_results.jsonl 
```