from client import chat
import sys, csv
sys.path.append('../common/')
from questions import questions
import pandas as pd

if __name__ == '__main__':
    counter = 1
    responses = []
    for q in questions:
        response = chat(q)
        if counter >= 2: # allow for a warmup
            responses.append(response)
        counter +=1

    df = pd.DataFrame(responses)
    df.to_csv('bench-bentoml.csv', index=False, quoting=csv.QUOTE_ALL)
