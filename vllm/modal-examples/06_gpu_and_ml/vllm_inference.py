# # Fast inference with vLLM (Llama 2 13B)
#
# In this example, we show how to run basic inference, using [`vLLM`](https://github.com/vllm-project/vllm)
# to take advantage of PagedAttention, which speeds up sequential inferences with optimized key-value caching.
#
# `vLLM` also supports a use case as a FastAPI server which we will explore in a future guide. This example
# walks through setting up an environment that works with `vLLM ` for basic inference.
#
# We are running the Llama 2 13B model here, and you can expect 30 second cold starts and well over 100 tokens/second.
# The larger the batch of prompts, the higher the throughput. For example, with the 60 prompts below,
# we can produce 24k tokens in 39 seconds, which is around 600 tokens/second.
#
# To run
# [any of the other supported models](https://vllm.readthedocs.io/en/latest/models/supported_models.html),
# simply replace the model name in the download step. You may also need to enable `trust_remote_code` for MPT models (see comment below)..
#
# ## Setup
#
# First we import the components we need from `modal`.

from modal import Stub, Image, Secret, method
import os


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Huggingface - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
#
# Since the weights are gated on HuggingFace, we must request access in two places:
# - on the [model card page](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
# - accept the license [on the Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).
#
# Next, [create a HuggingFace access token](https://huggingface.co/settings/tokens).
# To access the token in a Modal function, we can create a secret on the [secrets page](https://modal.com/secrets).
# Now the token will be available via the environment variable named `HUGGINGFACE_TOKEN`. Functions that inject this secret will have access to the environment variable.
#
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# Tip: avoid using global variables in this function. Changes to code outside this function will not be detected and the download step will not re-run.
def download_model_to_folder():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "meta-llama/Llama-2-13b-chat-hf",
        local_dir="/model",
        token=os.environ["HUGGINGFACE_TOKEN"],
    )



MODEL_DIR = "/model"

# ### Image definition
# We’ll start from a Dockerhub image recommended by `vLLM`, upgrade the older
# version of `torch` to a new one specifically built for CUDA 11.8. Next, we install `vLLM` from source to get the latest updates.
# Finally, we’ll use run_function to run the function defined above to ensure the weights of the model
# are saved within the container image.
#
image = (
    Image.from_dockerhub("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118"
    )
    # Pin vLLM to 07/19/2023
    .pip_install(
        "vllm @ git+https://github.com/vllm-project/vllm.git@bda41c70ddb124134935a90a0d51304d2ac035e8"
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder, secret=Secret.from_name("my-huggingface-secret")
    )
)

stub = Stub("example-vllm-inference", image=image)


# ## The model class
#
# The inference function is best represented with Modal's [class syntax](/docs/guide/lifecycle-functions) and the `__enter__` method.
# This enables us to load the model into memory just once every time a container starts up, and keep it cached
# on the GPU for each subsequent invocation of the function.
#
# The `vLLM` library allows the code to remain quite clean.
@stub.cls(gpu="A100", secret=Secret.from_name("my-huggingface-secret"))
class Model:
    def __enter__(self):
        from vllm import LLM

        # Load the model. Tip: MPT models may require `trust_remote_code=true`.
        self.llm = LLM(MODEL_DIR)
        self.template = """SYSTEM: You are a helpful assistant.
USER: {}
ASSISTANT: """

    @method()
    def generate(self, user_questions):
        from vllm import SamplingParams
        import time
        response = {'question': user_questions, 'note': 'vLLM/Modal A100'}

        prompts = user_questions
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1,
            max_tokens=200,
        )

        start = time.perf_counter()
        result = self.llm.generate(prompts, sampling_params)
        request_time = time.perf_counter() - start

        for output in result:
            response['tok_count'] = len(output.outputs[0].token_ids)
            response['time'] = request_time
            response['answer'] = output.outputs[0].text
        return response


# ## Run the model
# We define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run vllm_inference.py`.
@stub.local_entrypoint()
def main():
    import pandas as pd
    from tqdm import tqdm
    model = Model()
    questions = [
        # Coding questions
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
        "What are the differences between Javascript and Python?",
        # Literature
        "Write a story in the style of James Joyce about a trip to the Australian outback in 2083, to see robots in the beautiful desert.",
        "Who does Harry turn into a balloon?",
        "Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
        # Math
        "What is the product of 9 and 8?",
        "If a train travels 120 kilometers in 2 hours, what is its average speed?",
        "Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
    ]
    counter = 1
    responses = []
    for q in tqdm(questions):
        response = model.generate.call([q])
        if counter >= 2:
            responses.append(response)
        counter += 1

    df = pd.DataFrame(responses)
    df.to_csv('bench-vllm.csv', index=False)
