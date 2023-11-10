import os, requests, time, json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",
                                          use_auth_token=True)

api_base = os.getenv("OPENAI_API_BASE")
token = os.getenv("OPENAI_API_KEY")
def chat(prompt:str):
    payload = {"model": "meta-llama/Llama-2-7b-chat-hf",
               "messages": [{"role": "system", "content": "You are very verbose"},
                            {"role": "user", "content": f"{prompt}"}],
               "max_tokens":200}
    headers={"Authorization": f"Bearer {token}"}
    start = time.perf_counter()
    response = requests.post(f"{api_base}/chat/completions", 
                             headers=headers, 
                             json=payload)
    generated_text = response.json()['choices'][0]['message']['content']
    request_time = time.perf_counter() - start

    return {'tok_count': len(tokenizer.encode(generated_text)),
        'time': request_time,
        'question': prompt,
        'answer': generated_text,
        'note': 'anyscale'}

if __name__ == '__main__':
    prompt = "San Francisco is a city in"
    print(f"User: {prompt}\nLlama2: {chat(prompt)['answer']})")
