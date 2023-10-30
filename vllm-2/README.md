A refresh of the prior [vLLM Benchmark](../vllm/). 

## Setup

Install vllm per [these instructions](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html)

## Start Inference Server

Start the vLLM inference server with the following command:

```bash
python -m api_server --model meta-llama/Llama-2-7b-hf
```

## Test the infernce server

```bash
curl http://localhost:8000/generate \
    -d '{
        "prompt": "San Francisco is a",
        "max_tokens": 200
    }'
```

