#!/bin/bash

if [ -z "$HUGGING_FACE_HUB_TOKEN" ]
then
  echo "HUGGING_FACE_HUB_TOKEN is not set. Please set it before running this script."
  exit 1
fi

model=meta-llama/Llama-2-7b-hf
volume=$PWD/data 

docker run --gpus all \
-e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
 --shm-size 5g -p 8081:80 \
 -v $volume:/data ghcr.io/huggingface/text-generation-inference \
 --model-id $model \
 --max-best-of 1 "$@"
