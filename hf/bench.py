# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
import time
import sys
sys.path.append('../common/')
from questions import questions
import pandas as pd

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)


prompt = "Hello, my llama's name is Zach"

def predict(prompt:str):
    start_time = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = model_nf4.generate(**inputs, max_length=200)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    request_time = time.perf_counter() - start_time
    return {'tok_count': generated_ids.shape[1],
        'time': request_time,
        'question': prompt,
        'answer': output,
        'note': 'nf4 4bit quantization bitsandbytes'}


if __name__ == '__main__':
    counter = 1
    responses = []

    for q in questions:
        if counter >= 2: responses.append(predict(q))
        counter += 1

    df = pd.DataFrame(responses)
    df.to_csv('bench-hf.csv', index=False)

