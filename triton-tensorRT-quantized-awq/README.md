# Nvidia Triton w/ TensorRT-LLM Backend + AWQ Quantization

Use the [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main) backend with the [Nvidia Triton Inference Server](https://github.com/triton-inference-server/server).

The clearest end-to-end instructions I found was [this official blog post](https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/).

## Build TensorRT-LLM container

Follow [these instructions](https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.5.0/docs/source/installation.md) to build the docker container to compile the model.  

When you are done this will have created a docker image called `tensorrt_llm/release:latest ` locally.

> Note: I had to fight nvidia-docker for this to work, I ended up having to uninstall Docker and anything related to nvidia container toolkit and re-install everything from scratch.

## Pull the model from HuggingFace

Make a directory called model_input and clone the Hugging Face model into it.

```bash
mkdir model_input
# Make sure you have git-lfs installed (https://git-lfs.com)
cd model_input
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
```

## Compile the model

To compile the model, mount the model you just pulled from HuggingFace and the model_output directory into the container and run the compile script.  First, shell into the container like this:

```bash
# Make an output directory to store the compiled model assets
mkdir model_output

sudo docker run --gpus all -it -v ${PWD}/model_input:/model_input -v  ${PWD}/model_output:/model_output tensorrt_llm/release:latest bash
```

Install the quantization toolkit per [these instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0/examples/quantization#tensorrt-llm-quantization-toolkit-installation-guide):

```bash
cd /app/tensorrt_llm/examples/quantization
python -m pip install --upgrade pip
# Obtain the cuda version from the system. Assuming nvcc is available in path.
cuda_version=$(nvcc --version | grep 'release' | awk '{print $6}' | awk -F'[V.]' '{print $2$3}')
# Obtain the python version from the system.
python_version=$(python3 --version 2>&1 | awk '{print $2}' | awk -F. '{print $1$2}')
# Download and install the AMMO package from the DevZone.
wget https://developer.nvidia.com/downloads/assets/cuda/files/nvidia-ammo/nvidia_ammo-0.3.0.tar.gz
tar -xzf nvidia_ammo-0.3.0.tar.gz
pip install nvidia_ammo-0.3.0/nvidia_ammo-0.3.0+cu$cuda_version-cp$python_version-cp$python_version-linux_x86_64.whl
# Install the additional requirements
pip install -r requirements.txt
```

Then quantize the model, this took < 10 minutes on my RTX 6000 Ada (so be patient):

```bash
# Quantize HF LLaMA 7B checkpoint into INT4 AWQ format
cd /app/tensorrt_llm/examples/llama
python quantize.py --model_dir /model_input/Llama-2-7b-hf/ \
                --dtype float16 \
                --qformat int4_awq \
                --export_path ./llama-7b-4bit-gs128-awq.pt \
                --calib_size 32
```

Then, run the compile script.  Make sure your GPU memory is free when you do this:

```bash
cd /app/tensorrt_llm/examples/llama
# Compile the LLaMA 7B model to TensorRT format
python build.py --model_dir /model_input/Llama-2-7b-hf/ \
                --quant_ckpt_path ./llama-7b-4bit-gs128-awq.pt \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --remove_input_padding \
                --use_inflight_batching \
                --paged_kv_cache \
                --use_weight_only \
                --weight_only_precision int4_awq \
                --per_group \
                --output_dir /model_output/
```


When you are done, exit the docker container.  The compiled assets will be located in `model_output/`.  You will see three files:

- `llama_float16_tp1_rank0.engine`: The main output of the build script, containing the executable graph of operations with the model weights embedded.
- `config.json`: Includes detailed information about the model, like its general structure and precision, as well as information about which plug-ins were incorporated into the engine.
- `model.cache`: Caches some of the timing and optimization information from model compilation, making successive builds quicker.



## Prepare the model repository

The triton inference server works with model repositories that are specific directory structures with config files and other assets.  You can read about model repositories [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html).  The model repository for this example is quite complicated and involved setting up an ensemble of a preprocessing, model and postprocessing components along with lots of boilerplate code.  

The easiest way to get started is to clone the example repo and modify it to suit your needs.  First clone the the repo:

```bash
git clone -b release/0.5.0 https://github.com/triton-inference-server/tensorrtllm_backend.git
```

Copy the compiled model assets from `./model_output` into the model example repository:

```bash
cp model_output/* tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/1/
```

Then use their tools to modify the configuration files of all three components of the ensemble. Make sure you run these commands in the `tensorrtllm_backend` directory:

```bash
cd tensorrtllm_backend
# modify config for the model
python3 tools/fill_template.py --in_place \
      all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
      decoupled_mode:true,engine_dir:/all_models/inflight_batcher_llm/tensorrt_llm/1,\
max_tokens_in_paged_kv_cache:,batch_scheduler_policy:guaranteed_completion,kv_cache_free_gpu_mem_fraction:0.2,\
max_num_sequences:4
```

Next, modify config for the preprocessing component, modify the `tokenizer_dir` to point to a model on HuggingFace Hub you used, I am using `NousResearch/Llama-2-7b-hf` which is a replica of `meta-llama/Llama-2-7b-hf`, so we don't have to worry about the fiddly permissions on the original model.

```bash
# modify config for the preprocessing component
python tools/fill_template.py --in_place \
    all_models/inflight_batcher_llm/preprocessing/config.pbtxt \
    tokenizer_type:llama,tokenizer_dir:NousResearch/Llama-2-7b-hf

# modify config for the postprocessing component
python tools/fill_template.py --in_place \
    all_models/inflight_batcher_llm/postprocessing/config.pbtxt \
    tokenizer_type:llama,tokenizer_dir:NousResearch/Llama-2-7b-hf
```

## Prepare The Triton Server

Next, we have to mount the model repository we just created into the Triton server and do some additional work interactively before it is ready.  Make sure you are in the `tensorrtllm_backend` directory when running the following commands because we also need to mount the `scripts` directory into the container.

```bash
sudo docker run -it --rm --gpus all --network host --shm-size=1g \
-v $(pwd)/all_models:/all_models \
-v $(pwd)/scripts:/opt/scripts \
nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3 bash
```

Next, in the Docker container, login to the HuggingFace Hub:

Then, install the python dependencies:

```bash
# Install python dependencies
pip install sentencepiece protobuf
```

Finally, start the Triton server:

```bash
# Launch Server
python /opt/scripts/launch_triton_server.py --model_repo /all_models/inflight_batcher_llm --world_size 1
```

> Note: if you get an error `Unexpected tokenizer type: ${tokenizer_type}` this means you didn't run the `fill_template.py` script on the preprocessing and postprocessing config files correctly.

You will get output that looks like this:

```bash
I1101 14:59:56.742506 113 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
I1101 14:59:56.742703 113 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
I1101 14:59:56.828990 113 http_server.cc:270] Started Metrics Service at 0.0.0.0:8002
```

### Test the server

You can make a request with `curl` like this:

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d \
'{"text_input": "How do I count to nine in French?",
"parameters": {"max_tokens": 100, "bad_words":[""],"stop_words":[""]}}'
```
