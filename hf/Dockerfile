# From the parent directory (main directory of this repo) run:
#
# docker build --build-arg USERID=$(id -u) -t local/hf-bench hf
#
# If not already using and having a $HOME/.cache/huggingface/ then:
#
# mkdir $HOME/.cache/huggingface/
# docker run --rm -it -v$HOME/.cache/huggingface/:/home/user/.cache/huggingface/ local/hf-bench \
#   huggingface-cli login
# Answer n to: Add token as git credential? (Y/n) n
#
# docker run --rm -it -v$HOME/.cache/huggingface/:/home/user/.cache/huggingface/ \
#   -v$(pwd):/home/user/llama-inference --gpus all local/hf-bench \
#   sh -c 'cd /home/user/llama-inference/hf && python3 bench.py'
#
# You can substitute bench.py by bench-bb.py, bench-gptq.py or any other.
# If using Podman with CDI substitute
#   --gpus all
# for
#   --device nvidia.com/gpu=all --security-opt=label=disable

# Select an available version from
# https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md:
FROM nvidia/cuda:12.2.2-cudnn8-runtime-rockylinux9

RUN yum install -y \
  python3-pip cuda-cupti-$(echo $CUDA_VERSION | sed -r 's/(.+)[.](.+)[.].*/\1-\2/') && \
  yum clean all && rm -rf /var/cache/yum/*

# 2023-12-07 giving ImportError regarding bitsandbytes and accelerate unless using --pre, also for
# some reason some package needs scipy: 
RUN pip install --no-cache-dir --pre transformers accelerate optimum bitsandbytes auto_gptq scipy

# PyTorch doesn't find it by default, /usr/local/lib64/ doesn't work neither:
RUN ln -s /usr/local/cuda-12/lib64/libcupti.so.12 /usr/lib64/

ARG USERID=1000
RUN adduser -u $USERID user
USER user

