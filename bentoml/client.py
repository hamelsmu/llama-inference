import requests, json, time
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",
                                          use_auth_token=True)

url = "http://127.0.0.1:8000/v1/generate"
headers = {'Content-Type': 'application/json'}

def chat(prompt:str):
    data = {"prompt": prompt, 
            "llm_config": {
                "max_new_tokens": 200,
                "use_llama2_prompt": False,
                }
            }
    start = time.perf_counter()
    response = requests.post(url, headers=headers, data=json.dumps(data))
    generated_text = response.json()['responses'][0]
    request_time = time.perf_counter() - start
    return {'tok_count': len(tokenizer.encode(generated_text)),
            'time': request_time,
            'question': prompt,
            'answer': f"{generated_text}",
            'note': 'bentoml-vllm'}

if __name__ == '__main__':
    prompt = "San Francisco is a city in"
    print(f"User: {prompt}\nLlama2: {chat(prompt)['answer']})")
