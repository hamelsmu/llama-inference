# This is a failed experiment

This is the same as [this example](../triton-vllm-awq-4bit/) except we are going to build an 8-bit model.  `TheBloke/Llama-2-7B-AWQ` used in that example is a 4-bit model.

## Quantize the Model

To quantize the model we will use [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), and run [quantize.py](./quantize.py).


First install AutoAWQ and login to HuggingFace:

```bash
pip install autoawq
huggingface-cli login
```

Then run the quantization script:

```bash
python quantize.py
```

## YOU CANNOT DO THIS

**Only 4bit is supported for now:**

```
/home/hamel/mambaforge/envs/autoawq/lib/python3.10/site-packages/datasets/table.py:1421: FutureWarning: promote has been superseded by mode='default'.
  table = cls._concat_blocks(blocks, axis=0)
AWQ:   0%|                                                                                                                  | 0/32 [00:13<?, ?it/s]
Traceback (most recent call last):
  File "/home/hamel/github/llama-inference/triton-vllm-awq-8bit/quantize.py", line 13, in <module>
    model.quantize(tokenizer, quant_config=quant_config)
  File "/home/hamel/mambaforge/envs/autoawq/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/home/hamel/mambaforge/envs/autoawq/lib/python3.10/site-packages/awq/models/base.py", line 49, in quantize
    quantizer.quantize()
  File "/home/hamel/mambaforge/envs/autoawq/lib/python3.10/site-packages/awq/quantize/quantizer.py", line 79, in quantize
    self._apply_quant(self.modules[i], named_linears)
  File "/home/hamel/mambaforge/envs/autoawq/lib/python3.10/site-packages/awq/quantize/quantizer.py", line 100, in _apply_quant
    q_linear = q_linear_module.from_linear(
  File "/home/hamel/mambaforge/envs/autoawq/lib/python3.10/site-packages/awq/modules/linear.py", line 50, in from_linear
    awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
  File "/home/hamel/mambaforge/envs/autoawq/lib/python3.10/site-packages/awq/modules/linear.py", line 29, in __init__
    raise NotImplementedError("Only 4-bit are supported for now.")
NotImplementedError: Only 4-bit are supported for now.
```

