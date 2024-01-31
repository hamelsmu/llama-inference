#PowerInfer benchmark

Benchmark for [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer).

Note that the model loses some inference quality in exchange for speed as shown in https://huggingface.co/SparseLLM/ReluLLaMA-7B.

You can compile PowerInfer following their instructions and then run the server:

```bash
build/bin/server -v -m ReluLLaMA-7B/llama-7b-relu.powerinfer.gguf
```

And in another terminal run:
```bash
python3 bench.py
```

The results will be in the bench-powerinfer.csv file.

Or alternatively you can follow the instructions in the [Dockerfile](Dockerfile) to build a container to run the server and the benchmark inside it.

