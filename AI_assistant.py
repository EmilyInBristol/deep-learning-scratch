from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Baichuan 7B tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7b")
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/baichuan-7b")

# Generate text from a given prompt
prompt = "The future of artificial intelligence in China is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)

# Decode and print the generated text
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
