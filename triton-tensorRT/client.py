import requests, json, time
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",
                                          use_auth_token=True)

def chat(prompt:str):
    payload = {"text_input": prompt, "parameters": {"bad_words":[""],"stop_words":[""], "max_tokens": 200}}
    headers = {'Content-Type': 'application/json'}
    start = time.perf_counter()
    response = requests.post("localhost:8000/v2/models/ensemble/generate", headers=headers, data=json.dumps(payload))
    generated_text = response.json()["text_output"]
    request_time = time.perf_counter() - start

    return {'tok_count': len(tokenizer.encode(generated_text)),
        'time': request_time,
        'question': prompt,
        'answer': generated_text,
        'note': 'triton-vllm'}

if __name__ == '__main__':
    prompt = "San Francisco is a city in"
    print(f"User: {prompt}\nLlama2: {chat(prompt)['answer']})")
