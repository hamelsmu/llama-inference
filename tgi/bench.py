#see https://github.com/huggingface/text-generation-inference
import sys
import argparse
sys.path.append('../common/')
import time
import pandas as pd
from text_generation import Client
from questions import questions
from tqdm import tqdm


# Define the argument parser
parser = argparse.ArgumentParser(description='Run LLM inference requests and save to a csv.')
parser.add_argument('--filename', type=str, required=True, help='Path to the output CSV file.')
parser.add_argument('--note', type=str, required=True, help='Note to add to the rows of the file.')

# Parse the command-line arguments
args = parser.parse_args()

def generate_text_and_save_results(filename):

    client = Client("http://127.0.0.1:8081", timeout=120)
    counter = 1
    responses = []

    for q in tqdm(questions):
        start = time.perf_counter()
        result = client.generate(q, max_new_tokens=1000, best_of=1)
        request_time = time.perf_counter() - start
        if counter >= 2: # allow for a warmup
            responses.append({'tok_count': result.details.generated_tokens, 
                            'time': request_time,
                            'question': q, 
                            'answer': result.generated_text, 
                            'note': args.note})
        counter +=1

    df = pd.DataFrame(responses)
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    generate_text_and_save_results(args.filename)
