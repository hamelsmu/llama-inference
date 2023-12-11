# Load model directly
import time
import sys
sys.path.append('../common/')
from questions import questions
import pandas as pd

# Load model directly
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

model_name_or_path = "TheBloke/Llama-2-7B-GPTQ"
model_basename = "gptq-4bit-128g-actorder_True"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        revision=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)


def predict(prompt:str):
    start_time = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(**inputs, max_length=200)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    request_time = time.perf_counter() - start_time
    return {'tok_count': generated_ids.shape[1],
        'time': request_time,
        'question': prompt,
        'answer': output,
        'note': 'gptq'}


if __name__ == '__main__':
    counter = 1
    responses = []

    for q in questions:
        if counter >= 2: responses.append(predict(q))
        counter += 1

    df = pd.DataFrame(responses)
    df.to_csv('bench-hf-gptq.csv', index=False)

