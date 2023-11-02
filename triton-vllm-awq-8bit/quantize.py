from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'NousResearch/Llama-2-7b-hf'
quant_path = 'llama-2-7b-awq-8bit'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 8, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

model.push_to_hub(quant_path, use_auth_token=True)