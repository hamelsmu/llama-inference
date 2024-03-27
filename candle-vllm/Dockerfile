# syntax=docker/dockerfile:1

# From the parent directory (main directory of this repo) run:
#
# Note: if building on a machine with a different GPU or no GPU then check
# https://developer.nvidia.com/cuda-gpus and pass the value without the decimal point to
# CUDA_COMPUTE_CAP directly without the $(...), for example for an A100 is CUDA_COMPUTE_CAP=80 and
# for an A10 is CUDA_COMPUTE_CAP=86.
#
# docker build --build-arg USERID=$(id -u) --build-arg \
#   CUDA_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv | tail -n1 | tr -d .) \
#   -t local/candle-vllm-bench candle-vllm
#
# docker run --rm -it --name candle-vllm-bench \
#   -v$HOME/.cache/huggingface/:/home/user/.cache/huggingface/ \
#   -v$(pwd):/home/user/llama-inference --gpus all local/candle-vllm-bench \
#   --port 8080 llama7b --repeat-last-n 64
#
# In another terminal:
#
# docker exec -it -u0:0 candle-vllm-bench sh -c 'apt-get update && \
#   apt-get install -y python3-pip python3-requests python3-pandas && pip3 install transformers'
# docker exec -it candle-vllm-bench sh -c 'cd /home/user/llama-inference/anyscale && \
#   OPENAI_API_BASE=http://localhost:8080/v1 OPENAI_API_KEY=none python3 bench.py'
#
# If using Podman with CDI substitute
#   --gpus all
# for
#   --device nvidia.com/gpu=all --security-opt=label=disable

# Select an available version from
# https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md:
# Need to use older Ubuntu 22.04 instead of Rocky 9 or 8 because of linking issues
# https://github.com/huggingface/candle/issues/1844:
#FROM nvcr.io/nvidia/cuda:12.3.2-cudnn9-devel-rockylinux9 as build
FROM nvcr.io/nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04 as build

ARG CUDA_COMPUTE_CAP
#RUN dnf install -y cargo openssl-devel && dnf clean all && rm -rf /var/cache/dnf/*
RUN apt-get update && apt-get install -y curl libssl-dev pkg-config && rm -rf /var/lib/apt/lists/*
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && ln -s /root/.cargo/bin/* /usr/local/bin
ADD https://github.com/EricLBuehler/candle-vllm.git#master /candle-vllm
WORKDIR /candle-vllm
RUN cargo install --path . --features cuda,cudnn,flash-attn,nccl

#FROM nvcr.io/nvidia/cuda:12.3.2-cudnn9-runtime-rockylinux9
FROM nvcr.io/nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 as runtime
ARG USERID=1000
COPY --from=build /root/.cargo/bin/candle-vllm /usr/local/bin
RUN ln -s /usr/local/cuda/compat/libcuda.so.1 /lib/x86_64-linux-gnu/ && ldconfig
RUN adduser --disabled-password --gecos '' -u $USERID user
USER user
ENTRYPOINT ["/usr/local/bin/candle-vllm"]
CMD ["--port", "8080", "llama7b", "--repeat-last-n", "64"]

