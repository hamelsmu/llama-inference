# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.bettertransformer import BetterTransformer
import time
import sys
sys.path.append('../common/')
from questions import questions
import pandas as pd

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.to("cuda")
model = BetterTransformer.transform(model)


def predict(prompt:str):
    start_time = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        generated_ids = model.generate(**inputs, max_length=200)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    request_time = time.perf_counter() - start_time
    return {'tok_count': generated_ids.shape[1],
        'time': request_time,
        'question': prompt,
        'answer': output,
        'note': ''}


if __name__ == '__main__':
    counter = 1
    responses = []

    for q in questions:
        if counter >= 2: responses.append(predict(q))
        counter += 1

    df = pd.DataFrame(responses)
    df.to_csv('bench.csv', index=False)

