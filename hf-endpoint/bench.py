import requests, json, os, time
from transformers import AutoTokenizer
import sys
import pandas as pd
sys.path.append('../common/')
from questions import questions
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",
                                          use_auth_token=True)

url = "https://vbwx1aygju553ro6.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
    "Content-Type": "application/json",
}

def generate(prompt):
    data = {"inputs":prompt,
            "parameters": {'max_new_tokens': 200,
                        'return_full_text': False},
            "options": {'use_cache': False},
            }
    start = time.perf_counter()
    response = requests.post(url, headers=headers, data=json.dumps(data))
    request_time = time.perf_counter() - start
    return {'tok_count': len(tokenizer.encode(response.json()[0]['generated_text'])),
            'time': request_time,
            'question': prompt,
            'answer': response.json()[0]['generated_text'],
            'note': 'hf-endpoint'}

if __name__ == '__main__':
    counter = 1
    responses = []
    for q in tqdm(questions):
        response = generate(q)
        if counter >= 2: # allow for a warmup
            responses.append(response)
        counter +=1

    df = pd.DataFrame(responses)
    df.to_csv('bench-hf-endpoint.csv', index=False)
