import asyncio
import time
import aiohttp
import statistics

# Shared concurrency counter
current_concurrency = 0

async def send_request(session, url, data, request_number, response_record):
    global current_concurrency
    print(f"Starting request #{request_number}")
    current_concurrency += 1  # Increment concurrency when request starts
    start_time = time.perf_counter()

    async with session.post(url, json=data) as response:
        await response.read()

    end_time = time.perf_counter()
    latency = end_time - start_time
    response_record.append((current_concurrency, latency))
    print(f"Finished request #{request_number}")
    current_concurrency -= 1  # Decrement concurrency when request ends

async def main(duration, requests_per_second, output_seq_len):
    url = 'http://localhost:8000/v2/models/ensemble/generate'
    data = {
        "text_input": "How do I count to ten in French?",
        "parameters": {
            "max_tokens": output_seq_len,
            "min_length": output_seq_len,
            "bad_words": [""],
            "stop_words": ["</s>"],
            # "stream": True
        }
    }

    tasks = []
    response_record = []
    request_counter = 0

    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < duration:
            request_counter += 1
            task = asyncio.create_task(send_request(session, url, data, request_counter, response_record))
            tasks.append(task)
            await asyncio.sleep(1 / requests_per_second)
            print(f"Current concurrency: {current_concurrency}")

        await asyncio.gather(*tasks)

    # Statistics
    latencies = [item[1] for item in response_record]
    average_latency = statistics.mean(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)
    std_dev_latency = statistics.stdev(latencies)

    print(f"Average Latency: {average_latency:.4f} seconds")
    print(f"Max Latency: {max_latency:.4f} seconds")
    print(f"Min Latency: {min_latency:.4f} seconds")
    print(f"Standard Deviation of Latency: {std_dev_latency:.4f} seconds")



if __name__ == "__main__":
    duration = 60  # Duration in seconds
    requests_per_second = .3  # Requests per second
    output_seq_len = 300
    asyncio.run(main(duration, requests_per_second, output_seq_len))