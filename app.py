import json, time, uuid

from typing import List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = "openai-community/gpt2"

app = FastAPI()

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
    top_p: float = 1.0
    stream: bool = False

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = checkpoint
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False

def top_p_sampling(logits, top_p=0.9, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff = cumulative_probs > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False

    sorted_probs[cutoff] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, next_token)

def stream_response(prompt, max_tokens, temperature, top_p):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids

    for _ in range(max_tokens):
        with torch.no_grad():
            output = model(input_ids=generated)
        
        next_token_logits = output.logits[:, -1, :]
        next_token = top_p_sampling(next_token_logits, top_p, temperature)
        generated = torch.cat((generated, next_token), dim=1)

        decoded = tokenizer.decode(next_token[0], skip_special_tokens=True)

        yield f"data: {json.dumps({'choices': [{'text': decoded, 'index': 0}]})}\n\n"

    yield "data: [DONE]\n\n"

@app.post("/v1/completions")
async def completion(request: CompletionRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.shape[1]
    max_input_length = model.config.max_position_embeddings

    if prompt_length > max_input_length:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt is too long: {prompt_length} tokens. Model supports up to {max_input_length} tokens."
        )
    
    if request.stream:
        return StreamingResponse(
            stream_response(
                request.prompt,
                request.max_tokens,
                request.temperature,
                request.top_p
            ),
            media_type="text/event-stream"
        )
    
    inputs = tokenizer(request.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    }

def format_messages(messages: List[ChatMessage]) -> str:
    prompt = ""

    for msg in messages:
        if msg.role == "system":
            prompt += f"<<SYS>> {msg.content}\n"

        elif msg.role == "user":
            prompt += f"[User]: {msg.content}\n"

        elif msg.role == "assistant":
            prompt += f"[Assistant]: {msg.content}\n"

    prompt += "[Assistant]: "
    
    return prompt

def stream_chat_response(prompt, max_tokens, temperature, top_p):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids

    for _ in range(max_tokens):
        with torch.no_grad():
            output = model(input_ids=generated)
        
        next_token_logits = output.logits[:, -1, :]
        next_token = top_p_sampling(next_token_logits, top_p, temperature)
        generated = torch.cat((generated, next_token), dim=1)

        decoded = tokenizer.decode(next_token[0], skip_special_tokens=True)

        yield f"data: {json.dumps({'choices': [{'delta': {'content': decoded}, 'index': 0, 'finish_reason': None}]})}\n\n"

    yield "data: {\"choices\": [{\"delta\": {}, \"finish_reason\": \"length\", \"index\": 0}]}\n\n"

    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    prompt = format_messages(request.messages)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.shape[1]
    max_input_length = model.config.max_position_embeddings

    if prompt_length > max_input_length:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long: {prompt_length} tokens. Max: {max_input_length}"
        )

    if request.stream:
        return StreamingResponse(
            stream_chat_response(
                prompt,
                request.max_tokens,
                request.temperature,
                request.top_p,
            ),
            media_type="text/event-stream"
        )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion_text = text[len(prompt):]

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion_text},
                "finish_reason": "length"
            }
        ],
        "usage": {
            "prompt_tokens": inputs.input_ids.shape[1],
            "completion_tokens": request.max_tokens,
            "total_tokens": inputs.input_ids.shape[1] + request.max_tokens
        },
    }
