## Setup

1. Install the [`aws cli`](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
2. Install other dependencies with `pip install -U sagemaker boto3`
3. Run `aws configure`
4. Enter the `AWS Access Key ID` and `AWS Secret Access Key` given to you by your administrator.  Make sure they give you permissions to deploy sagemaker endpoints.

## Deploy Model

Look at the code in [`deploy.py`](./deploy.py), and modify any values that need to be changed, particularly `MODEL`, `MODEL_TO_EC2_TYPE`, `MODEL_TO_N_GPU` and `SM_EXEC_ROLE`.  Your AWS administrator can provide you with the value for `SM_EXEC_ROLE` which is the [SageMaker Execution Role](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html). For guidance on the type of hardware you need for various types of models, see the [Hardware requirements section of this blog post](https://www.philschmid.de/sagemaker-llama-llm).


One "must do" optimization is to enable flash attention which will provide an enormous speedup.  You can do this by setting the environment variable `USE_FLASH_ATTENTION=TRUE`, which you can see in the `config` variable in [`deploy.py`](./deploy.py). This allows the underlying [Text-Generation-Inference server](https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/utils/flash_attn.py) to use flash attention.

Finally, deploy the endpoint:

```bash
python deploy.py
```

## Inference

Replace the endpoint name with the endpoint name you deployed in [`client.py`](./client.py):

```diff
def chat_iter(
    prompt:str,
-   endpoint_name: str = "huggingface-pytorch-tgi-inference-2023-10-29-05-31-01-597",
+   endpoint_name: str = "<YOUR ENDPOINT NAME>",
    max_new_tokens = 200,
```

Then run [`client.py`](./client.py):

```bash
python deploy.py
```
