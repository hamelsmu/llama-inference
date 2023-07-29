import openai
import os, sys, time, argparse
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("Must set environment variable OPENAI_API_KEY")
import pandas as pd
from tqdm import tqdm

url = 'http://0.0.0.0:5001/v1'
openai.api_base = url
# Parse the command-line arguments
# Define the argument parser
parser = argparse.ArgumentParser(description='Run LLM inference requests and save to a csv.')
parser.add_argument('--filename', type=str, required=True, help='Path to the output CSV file.')
parser.add_argument('--note', type=str, required=True, help='Note to add to the rows of the file.')
args = parser.parse_args()


def generate_text_and_save_results(filename):
    counter = 1
    responses = []

    for q in tqdm(questions):
        start = time.perf_counter()
        result =openai.Completion.create(model='TheBloke_Llama-2-7B-GPTQ',
                                         prompt="Say this is a test",
                                         max_tokens=200,
                                         temperature=0)
        request_time = time.perf_counter() - start
        if counter >= 2: # allow for a warmup
            responses.append({'tok_count': result.usage.completion_tokens, 
                            'time': request_time,
                            'question': q, 
                            'answer': result.choices[0].text, 
                            'note': args.note})
        counter +=1

    df = pd.DataFrame(responses)
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    generate_text_and_save_results(args.filename)
