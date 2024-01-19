# From the parent directory (main directory of this repo) run:
#
# docker build --build-arg USERID=$(id -u) -t local/ctranslate-bench ctranslate
#
# If not already using and having a $HOME/.cache/huggingface/ then:
#
# mkdir $HOME/.cache/huggingface/
# docker run --rm -it -v$HOME/.cache/huggingface/:/home/user/.cache/huggingface/:Z \
#   local/ctranslate-bench huggingface-cli login
# Answer n to: Add token as git credential? (Y/n) n
#
# Needs 15G RAM free:
# docker run --rm -it -v$HOME/.cache/huggingface/:/home/user/.cache/huggingface/:Z \
#   -v$(pwd):/home/user/llama-inference:Z local/ctranslate-bench \
#   sh -c 'cd /home/user/llama-inference/ctranslate && ct2-transformers-converter \
#   --model meta-llama/Llama-2-7b-hf --output_dir llama-2-7b-ct2'
#
# docker run --rm -it -v$HOME/.cache/huggingface/:/home/user/.cache/huggingface/:Z \
#   -v$(pwd):/home/user/llama-inference:Z --gpus all local/ctranslate-bench \
#   sh -c 'cd /home/user/llama-inference/ctranslate && python3 bench.py'
#
# If using Podman with CDI substitute
#   --gpus all
# for
#   --device nvidia.com/gpu=all --security-opt=label=disable

# Select an available version from
# https://github.com/OpenNMT/CTranslate2/pkgs/container/ctranslate2:
FROM ghcr.io/opennmt/ctranslate2:latest-ubuntu20.04-cuda11.2

RUN pip install --no-cache-dir torch transformers pandas

# The parent image ENTRYPOINT does more harm than good, so disable it:
ENTRYPOINT []

ARG USERID=1000
RUN adduser --disabled-password --gecos '' -u $USERID user
USER user

