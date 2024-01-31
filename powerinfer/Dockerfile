# From the parent directory (main directory of this repo) run:
#
# docker build --build-arg USERID=$(id -u) -t local/powerinfer-bench powerinfer
#
# mkdir $HOME/.cache/{huggingface,PowerInfer}
# docker run --rm -it -v$HOME/.cache/huggingface/:/home/user/.cache/huggingface/:Z \
#   -v$HOME/.cache/PowerInfer/:/home/user/.cache/PowerInfer/:Z \
#   local/powerinfer-bench \
#   sh -c 'cd /home/user/.cache/PowerInfer/ && \
#   huggingface-cli download --resume-download --local-dir ReluLLaMA-7B \
#   --local-dir-use-symlinks False PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF'
#
# docker run --rm -it --gpus all -v$HOME/.cache/PowerInfer/:/home/user/.cache/PowerInfer/:Z \
#   local/powerinfer-bench \
#   build/bin/main --log-disable -m \
#   /home/user/.cache/PowerInfer/ReluLLaMA-7B/llama-7b-relu.powerinfer.gguf \
#   -p "Once upon a time"
#
# docker run --rm -it --name powerinfer-bench --gpus all \
#   -v$HOME/.cache/PowerInfer/:/home/user/.cache/PowerInfer/:Z \
#   -v$(pwd):/home/user/llama-inference/:Z \
#   local/powerinfer-bench \
#   build/bin/server -v -m \
#   /home/user/.cache/PowerInfer/ReluLLaMA-7B/llama-7b-relu.powerinfer.gguf
#
# In another terminal:
#
# docker exec -it powerinfer-bench curl --request POST --url http://localhost:8080/completion \
#   --header "Content-Type: application/json" --data \
#   '{"prompt": "Building a website can be done in 10 simple steps:", "n_predict": 128}'
#
# docker exec -it powerinfer-bench sh -c 'cd /home/user/llama-inference/powerinfer && \
#   python3 bench.py'
#
# If using Podman with CDI substitute
#   --gpus all
# for
#   --device nvidia.com/gpu=all --security-opt=label=disable

# Select an available version from
# https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md:
FROM nvidia/cuda:12.3.1-devel-rockylinux9
ARG USERID=1000
RUN yum install -y python3-pip cmake libcudnn8 git && yum clean all && rm -rf /var/cache/yum/*
RUN git clone https://github.com/SJTU-IPADS/PowerInfer
WORKDIR /PowerInfer
RUN pip install --no-cache-dir -r requirements.txt
RUN cmake -S . -B build -DLLAMA_CUBLAS=ON
RUN cmake --build build --config Release -j "$(nproc)"
RUN pip install --no-cache-dir pandas #for the benchmark.
RUN adduser -u $USERID user
USER user

