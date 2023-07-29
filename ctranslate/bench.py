import time
import ctranslate2
import transformers
import sys
sys.path.append('../common/')
from questions import questions
import pandas as pd

generator = ctranslate2.Generator("llama-2-7b-ct2", device="cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def predict(prompt:str):
    "Generate text give a prompt"
    start = time.perf_counter()
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
    results = generator.generate_batch([tokens], sampling_topk=1, max_length=200, include_prompt_in_result=False)
    tokens = results[0].sequences_ids[0]
    output = tokenizer.decode(tokens)
    request_time = time.perf_counter() - start
    return {'tok_count': len(tokens),
            'time': request_time,
            'question': prompt,
            'answer': output,
            'note': 'CTranslate2'}

if __name__ == '__main__':
    counter = 1
    responses = []

    for q in questions:
        if counter >= 2: responses.append(predict(q))
        counter += 1

    df = pd.DataFrame(responses)
    df.to_csv('bench-ctranslate.csv', index=False)
