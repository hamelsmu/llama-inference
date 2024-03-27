import time

import requests

api_base = "http://localhost:8080"


def chat(prompt: str):
    payload = {"prompt": f"{prompt}", "n_predict": 200}
    headers = {"Content-Type": "application/json"}
    start = time.perf_counter()
    response = requests.post(f"{api_base}/completion", headers=headers, json=payload)
    response = response.json()
    request_time = time.perf_counter() - start

    return {
        "tok_count": response["tokens_predicted"],
        "time": request_time,
        "question": prompt,
        "answer": response["content"],
        "note": "PowerInfer " + response["model"],
    }


if __name__ == "__main__":
    prompt = "San Francisco is a city in"
    print(f"User: {prompt}\nPowerInfer: {chat(prompt)['answer']}")