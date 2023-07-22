# llama inference

Exploration of throughput on various setups of inference with llama.

For my experiments, I will be using [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf).

I'm exploring answers to these questions:

- [] How does latency differ on a 3090 vs. A600 vs. A10 vs. A100?
- [] What is the latency of inference out of the box with transformers?  
- [] What is the latency when using various tools that speed up inference?  There are tools like [TGI](https://github.com/huggingface/text-generation-inference), [vLLM](https://github.com/vllm-project), [CTranslate](https://github.com/OpenNMT/CTranslate2) but also model-level optimizations like [4/8 bit quantization](https://twitter.com/joao_gante/status/1681593614676426753?s=20) (which some of the aforementioned tools can apply for you.)

## Caveats

- I didn't explore throughput.  That is a deep rabbit hole - I was just exploring latency for a single request.  You can tradeoff throughput and latency with various forms of batching requests.  
- I used default parameters for most tools.  The default parameters for these tools may not all be optimizing for the same thing (ex, TGI generates 2 samples and selects the best one by default.)  The key is trying to get a rough idea for different tools.
