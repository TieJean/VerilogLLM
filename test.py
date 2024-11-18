from utils.models import *
from utils.dataloader import codegen_template
from peft import PeftModel

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, tokenizer = load_codegen(device)
    model = PeftModel.from_pretrained(base_model, "./results/verilogLLM/")
    while True:
        prompt = input("Give me a task (Enter 'q' to exit): ")
        if prompt == "q":
            exit(0)
        prompt = codegen_template(prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # Generate code
        output = model.generate(
            inputs["input_ids"],
            max_length=1024,        # Maximum length of the output
            num_beams=5,           # Beam search for better quality
            temperature=0.7,       # Sampling temperature for diversity
            top_k=50,              # Limit sampling to top k tokens
            do_sample=True,
            top_p=0.95,            # Nucleus sampling for diversity
            repetition_penalty=1.2, # Penalize repetitive text
        )

        # Decode and print the generated code
        generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
        import pdb; pdb.set_trace()
        print(generated_code)
