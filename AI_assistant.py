from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model with trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/baichuan-7b", trust_remote_code=True)

# Continue with text generation or other tasks
prompt = "The future of artificial intelligence in China is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
