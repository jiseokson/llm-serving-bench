import json, time, uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

checkpoint = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

class CompletionRequest(BaseModel):
    model: str = checkpoint
    prompt: str
    max_tokens: int = 50
    temperature: float = 1.0
    stream: bool = False

def stream_response(prompt, max_tokens):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    generated = input_ids

    for _ in range(max_tokens):
        with torch.no_grad():
            output = model(input_ids=generated)

        next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)

        decoded = tokenizer.decode(next_token[0], skip_special_tokens=True)

        yield f"data: {json.dumps( {'choices': [{'text': decoded, 'index': 0}]} )}\n\n"

    yield "data: [DONE]\n\n"

@app.post("/generate")
async def completion(request: CompletionRequest):
    start = time.time()

    if request.stream:
        return StreamingResponse(
            stream_response(request.prompt, request.max_tokens),
            media_type="text/event-stream")
    
    inputs = tokenizer(request.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=request.max_tokens)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    elapsed_ms = round((time.time() - start) * 1000)

    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"text": text, "index": 0, "finish_reason": "length"}],
        "usage": {
            "prompt_tokens": inputs.input_ids.shape[1],
            "completion_tokens": request.max_tokens,
            "total_tokens": inputs.input_ids.shape[1] + request.max_tokens
        },
        "elapsed_ms": elapsed_ms
    }
