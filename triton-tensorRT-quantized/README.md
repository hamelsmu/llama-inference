# Nvidia Triton w/ TensorRT-LLM Backend + Quantization

This is the same as [this example](../triton-tensorRT/), except we build the model with one additional flag: `--use_weight_only`:

```diff
cd examples/llama
python build.py --model_dir /model_input/Llama-2-7b-hf/ \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --remove_input_padding \
                --use_inflight_batching \
                --paged_kv_cache \
+                --use_weight_only \
                --output_dir /model_output/
```

After you re-build make sure you copy the new model to the model repository:

```bash
cp model_output/* tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/1/
```