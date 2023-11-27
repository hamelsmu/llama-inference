#!/bin/bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
sudo apt-get update && sudo apt-get -y install git git-lfs

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs install
git lfs pull


# See https://developer.nvidia.com/cuda-gpus#compute to find out which version
# I'm using a A100 for this particular setup so that is `80-real`
make -C docker release_build CUDA_ARCHS="80-real"

cd ..
mkdir model_input
# Make sure you have git-lfs installed (https://git-lfs.com)
cd model_input
git clone https://huggingface.co/NousResearch/Llama-2-70b-chat-hf
