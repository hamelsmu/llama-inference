# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# model = model.to_bettertransformer()
model.to("cuda")

prompt = "Hello, my llama's name is Zach"

start_time = time.perf_counter()

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs)

request_time = time.perf_counter() - start_time
print(request_time)

start_time = time.perf_counter()
inputs = tokenizer("This is a diffedrent prompt", return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs)

request_time = time.perf_counter() - start_time

print(f'Final request time: {request_time}')
