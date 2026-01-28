import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from bitsandbytes import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-3B",
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(
    base_model,
    "llama3b-lora"   # your downloaded folder
)
tokenizer = AutoTokenizer.from_pretrained("llama3b-lora")
tokenizer.pad_token = tokenizer.eos_token
prompt = "Explain transformers in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7,
    top_p=0.9,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
