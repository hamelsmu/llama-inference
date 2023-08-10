from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout
from transformers import AutoTokenizer
import time
import sys
sys.path.append('../../common/')
from questions import questions
import pandas as pd



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
cm = ChatModule(model="Llama-2-7b-chat-hf-q4f16_1")

def tok_count(prompt:str):
    inputs = tokenizer(prompt)
    return len(inputs['input_ids'])

def predict(prompt:str):
    start_time = time.perf_counter()
    output = cm.generate(prompt=prompt)
    request_time = time.perf_counter() - start_time

    return {'tok_count': tok_count(prompt),
            'time': request_time,
            'question': prompt,
            'answer': output,
            'note': 'mlc chat 7b q4f16_1'}

if __name__ == '__main__':
    counter = 1
    responses = []

    for q in questions:
        if counter >= 2: responses.append(predict(q))
        counter += 1

    df = pd.DataFrame(responses)
    df.to_csv('bench-mlc.csv', index=False)
