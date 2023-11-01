
# Triton with vLLM Backend

## Pull the Triton Server with vLLM backend

```bash
docker pull nvcr.io/nvidia/tritonserver:23.10-vllm-python-py3
```

## Setup the Model Repository

See this [README](https://github.com/triton-inference-server/vllm_backend) for documentation on how to configure the model repository.

In this situation, we can configure the model by setting the appropriate values [`model.json`](./model_repository/vllm_model/1/model.json) file.  The repository also contains a config.pbtxt file that which I copied [from here](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/model_repository/vllm_model/config.pbtxt).


## Run the Triton Server

```bash
docker run --gpus=1 --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.10-vllm-python-py3 tritonserver --model-repository=/models
```

You will get output that looks like this

```
I1031 22:17:26.597031 1 metrics.cc:817] Collecting metrics for GPU 0: NVIDIA RTX 6000 Ada Generation
I1031 22:17:26.601937 1 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
I1031 22:17:26.602140 1 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
I1031 22:17:26.644132 1 http_server.cc:270] Started Metrics Service at 0.0.0.0:8002
```

In this case, the server is running on port 8000.

## Test the Triton Server

```bash
curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0, "max_tokens": 200}}'
```

## Run the benchmark

```bash
python bench.py
```
