
# Triton with vLLM Backend + AWQ

This is exactly the same as [this example](../triton-vllm/) except we are using AWQ quantized model which means changing te file `model_repository/vllm_model/1/model.json`:

```
{
    "model":"TheBloke/Llama-2-7B-AWQ",
    "disable_log_requests": "true",
    "quantization": "awq"
}
```
