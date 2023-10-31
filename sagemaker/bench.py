from client import chat_iter
import sys
sys.path.append('../common/')
from questions import questions
import pandas as pd
# from tqdm import tqdm


if __name__ == '__main__':
    counter = 1
    responses = []
    for q in questions:
        response = chat_iter(q)
        if counter >= 2: # allow for a warmup
            responses.append(response)
        counter +=1

    df = pd.DataFrame(responses)
    df.to_csv('bench-sagemaker-flashattn.csv', index=False)