# From the parent directory (main directory of this repo) run:
#
# docker build --build-arg USERID=$(id -u) -t local/mlc-bench mlc
#
# If not already using and having a $HOME/.cache/huggingface/ then:
#
# mkdir $HOME/.cache/huggingface/
# docker run --rm -it -v$HOME/.cache/huggingface/:/home/user/.cache/huggingface/:Z local/hf-bench \
#   huggingface-cli login
# Answer n to: Add token as git credential? (Y/n) n
#
# docker run --rm -it -v$HOME/.cache/huggingface/:/home/user/.cache/huggingface/:Z local/hf-bench \
#   huggingface-cli download meta-llama/Llama-2-7b-hf
#
# mkdir $HOME/.cache/mlc_chat/
# docker run --rm -it -v$HOME/.cache/huggingface/:/home/user/.cache/huggingface/:Z \
#   -v$HOME/.cache/mlc_chat/:/home/user/.cache/mlc_chat/:Z -v$(pwd):/home/user/llama-inference \
#   --gpus all local/mlc-bench \
#   sh -c 'cd /home/user/llama-inference/mlc && mlc_chat convert_weight --quantization q4f16_1 \
#   -o Llama-2-7b-hf-q4f16_1 ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/*'
#
# docker run --rm -it -v$HOME/.cache/huggingface/:/home/user/.cache/huggingface/:Z \
#   -v$HOME/.cache/mlc_chat/:/home/user/.cache/mlc_chat/:Z -v$(pwd):/home/user/llama-inference \
#   --gpus all local/mlc-bench \
#   sh -c 'cd /home/user/llama-inference/mlc && mlc_chat gen_config --quantization q4f16_1 \
#   --conv-template LM -o Llama-2-7b-hf-q4f16_1 \
#   ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/*'
#
# docker run --rm -it -v$HOME/.cache/huggingface/:/home/user/.cache/huggingface/:Z \
#   -v$HOME/.cache/mlc_chat/:/home/user/.cache/mlc_chat/:Z -v$(pwd):/home/user/llama-inference \
#   --gpus all local/mlc-bench \
#   sh -c 'cd /home/user/llama-inference/mlc && python3 mlc.py'
#
# If using Podman with CDI substitute
#   --gpus all
# for
#   --device nvidia.com/gpu=all --security-opt=label=disable

# Select an available version from
# https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md:
FROM nvidia/cuda:12.2.2-cudnn8-devel-rockylinux9

# requests isn't declared as a dependency by mlc-chat:
RUN yum install -y python3-pip python3-requests && yum clean all && rm -rf /var/cache/yum/*

# Don't know why libcuda.so.1 isn't on another place or configured in ld.so.conf.d,
# /usr/local/lib64/ doesn't work neither; also PyTorch doesn't find libcupti.so.12 by default:
RUN ln -s /usr/local/cuda-12.2/compat/libcuda.so.1 /usr/lib64/ && \
  ln -s /usr/local/cuda-12/lib64/libcupti.so.12 /usr/lib64/

# Select a matching version from https://llm.mlc.ai/docs/install/mlc_llm.html#install-mlc-packages,
# also torch isn't declared as dependency by mlc-chat and it's required for convert_weight command:
RUN pip install --no-cache-dir --pre -f https://mlc.ai/wheels mlc-chat-nightly-cu122 \
    mlc-ai-nightly-cu122 transformers torch pandas

ARG USERID=1000
RUN adduser -u $USERID user
USER user

