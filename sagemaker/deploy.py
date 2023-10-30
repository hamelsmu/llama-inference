# Heavily inspired by: https://www.philschmid.de/sagemaker-llama-llm

import os
import json
import boto3
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.huggingface import HuggingFaceModel

MODEL = "7b"
MODEL_TO_EC2_TYPE = {"7b": "ml.g5.2xlarge", "13b": "ml.g5.12xlarge", "70b": "ml.p4d.24xlarge"}
MODEL_TO_N_GPU = {"7b": 1, "13b": 4, "70b": 8}
SM_EXEC_ROLE = 'llama2-ob-hamel-project'

config = {
  'HF_MODEL_ID': "meta-llama/Llama-2-%s-hf" % MODEL,
  'SM_NUM_GPUS': json.dumps(MODEL_TO_N_GPU[MODEL]),
  'MAX_INPUT_LENGTH': json.dumps(2048),
  'MAX_TOTAL_TOKENS': json.dumps(4096),
  'MAX_BATCH_TOTAL_TOKENS': json.dumps(8192),
  'HUGGING_FACE_HUB_TOKEN': os.environ['HUGGING_FACE_HUB_TOKEN']
}

def main(health_check_timeout=600):
    iam = boto3.client('iam')
    role = iam.get_role(RoleName=SM_EXEC_ROLE)['Role']['Arn']
    image = get_huggingface_llm_image_uri("huggingface")
    print(f"LLM image uri: {image}")
    llm_model = HuggingFaceModel(role=role, image_uri=image, env=config)
    _ = llm_model.deploy(
        initial_instance_count=1,
        instance_type=MODEL_TO_EC2_TYPE[MODEL],
        container_startup_health_check_timeout=health_check_timeout
    )

if __name__ == "__main__":
    main()