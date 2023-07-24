#!/bin/bash

if [ -z "$HUGGING_FACE_HUB_TOKEN" ]
then
  echo "HUGGING_FACE_HUB_TOKEN is not set. Please set it before running this script."
  exit 1
fi

volume=$PWD/data 

docker run --gpus all \
 -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
 -e GPTQ_BITS=4 -e GPTQ_GROUPSIZE=128 \
 --shm-size 5g -p 8081:80 \
 -v $volume:/data ghcr.io/huggingface/text-generation-inference \
 --max-best-of 1 "$@"
