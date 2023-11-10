To run this benchmark, you need to set the following environment variables:

```bash
export OPENAI_API_BASE="https://api.endpoints.anyscale.com/v1"
export OPENAI_API_KEY="YOUR_ANYSCALE_ENDPOINT_API_KEY"
```


You can test that the endpoint is running like so:

```
curl "$OPENAI_API_BASE/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'
```

You can run the benchmark like so:

```bash
python bency.py
```