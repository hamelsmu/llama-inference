I deployed an [inference endpoint](https://ui.endpoints.huggingface.co/) on HuggingFace for [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf), on a `Nvidia A10G` GPU.

I didn't try to turn on any optimizations like quantization and wanted to see what the default performance would be like.


The documentation for these interfaces can be found [here](https://huggingface.github.io/text-generation-inference/#/).  There is also a python client for this [here](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client#huggingface_hub.InferenceClient.text_generation).


