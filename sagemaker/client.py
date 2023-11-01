from sagemaker.huggingface.model import HuggingFacePredictor
from transformers import AutoTokenizer
import time

# Heavily inspired by: https://www.philschmid.de/sagemaker-llama-llm
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",
                                          use_auth_token=True)


def chat_iter(
    prompt:str,
    endpoint_name: str = "huggingface-pytorch-tgi-inference-2023-11-01-03-21-56-541",
    max_new_tokens = 200,
):
    llm = HuggingFacePredictor(endpoint_name=endpoint_name)
    payload = {
        "inputs":  prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "stop": ["</s>"]
        }
    }
    start = time.perf_counter()
    response = llm.predict(payload)
    generated_text = response[0]["generated_text"]
    request_time = time.perf_counter() - start

    return {'tok_count': len(tokenizer.encode(generated_text)),
        'time': request_time,
        'question': prompt,
        'answer': generated_text,
        'note': 'sagemaker-realtime-hf-endpoint-flashattention'}

def main():
    # example 1
    msg = "Make a recipe to make a very unique cake."
    response = chat_iter(msg)
    print(f"User: {msg}\nLlama2: {response['answer']}")

    # example 2
    msg = "How does FSDP work? What do I need to do to get started?"
    response = chat_iter(msg)
    print(f"User: {msg}\nLlama2: {response['answer']}")

if __name__ == '__main__':
    main()
