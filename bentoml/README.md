For bentoml, we will investigate [OpenLLM](https://github.com/bentoml/OpenLLM).  BentoML is a frontend that has various backends, like vLLM.  We will use the vLLM endpoint.

## Setup

```bash
pip install "openllm[llama, vllm]"
```

## Start The Server

```bash
export OPENLLM_ENDPOINT=http://localhost:5701
openllm start llama --model-id meta-llama/Llama-2-7b-hf --backend vllm
```

## Test The Seerver

```bash
curl -i http://127.0.0.1:8000/readyz
```


## Run The Benchmark

```bash
python bench.py
```
