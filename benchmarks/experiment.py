import os
import sys
import json
import asyncio
import argparse

import httpx

task2dataset = {
    "chat": "oasst1",
    "code": "humaneval",
    "qa": "triviaqa",
    "summarization": "cnn",
}

model2apiver = {
    "gpt2": "completion",
    "phi2": "completion",
    "llama2": "chat",
    "mistral": "chat",
}

apiver2endpoint = {
    "completion": "/v1/completions",
    "chat": "/v1/chat/completions",
}

model2id = {
    "gpt2": "openai-community/gpt2",
    "phi2": "microsoft/phi-2",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}

generation_config = {
    ("gpt2", "chat"): {"temperature": 0.9, "top_p": 0.95, "max_tokens": 100},
    ("gpt2", "summarization"): {"temperature": 0.5, "top_p": 0.9, "max_tokens": 60},
    ("gpt2", "qa"): {"temperature": 0.3, "top_p": 1.0, "max_tokens": 40},
    ("gpt2", "code"): {"temperature": 0.8, "top_p": 0.9, "max_tokens": 100},

    ("phi2", "chat"): {"temperature": 0.8, "top_p": 0.9, "max_tokens": 100},
    ("phi2", "summarization"): {"temperature": 0.4, "top_p": 0.9, "max_tokens": 60},
    ("phi2", "qa"): {"temperature": 0.2, "top_p": 1.0, "max_tokens": 40},
    ("phi2", "code"): {"temperature": 0.7, "top_p": 0.9, "max_tokens": 90},

    ("llama2", "chat"): {"temperature": 0.7, "top_p": 0.95, "max_tokens": 150},
    ("llama2", "summarization"): {"temperature": 0.3, "top_p": 0.9, "max_tokens": 80},
    ("llama2", "qa"): {"temperature": 0.1, "top_p": 1.0, "max_tokens": 50},
    ("llama2", "code"): {"temperature": 0.6, "top_p": 0.85, "max_tokens": 120},

    ("mistral", "chat"): {"temperature": 0.8, "top_p": 0.95, "max_tokens": 120},
    ("mistral", "summarization"): {"temperature": 0.3, "top_p": 0.9, "max_tokens": 70},
    ("mistral", "qa"): {"temperature": 0.2, "top_p": 1.0, "max_tokens": 50},
    ("mistral", "code"): {"temperature": 0.7, "top_p": 0.9, "max_tokens": 110},
}

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Serving Benchmark")

    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name. One of: gpt2, phi2, llama2, mistral")
    parser.add_argument(
        "--task", type=str, required=True,
        help="Task type. One of: chat, summarization, qa, codegen")
    parser.add_argument(
        "--mode", type=str, choices=["sync", "stream"], required=True,
        help="API request mode. Choose between \"sync\" or \"stream\"")
    parser.add_argument(
        "--repeat", type=int, required=True,
        help="Number of repetitions for the prompt set")

    parser.add_argument(
        "--throughput-only", action="store_true",
        help="Run in throughput-only mode (latency-related metrics will not be measured)")
    parser.add_argument(
        "--parallel", type=int, default=None,
        help="Number of concurrent requests. Required when using --throughput-only")

    args = parser.parse_args()

    if args.throughput_only and args.parallel is None:
        print("[Error] --parallel must be specified when --throughput-only is set", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    return args

def prompts(model, task):
    prompt_path = os.path.join("prompts", f"{task}_{task2dataset[task]}_{model2apiver[model]}.jsonl")

    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

class CompletionRequest:
    def __init__(self, model, task, prompt, mode):
        self.model = model
        self.taks = task
        self.prompt = prompt
        self.mode = mode

        self.model_id = model2id[model]
        self.apiver = model2apiver[model]

        self.endpoint = host + apiver2endpoint[self.apiver]

        self.generation_args = generation_config[(model, task)]

        if self.apiver == "completion":
            self.payload = {
                "model": self.model_id,
                "stream": True if mode == "stream" else False,
                **prompt,
                **self.generation_args}

        elif self.apiver == "chat":
            self.payload = {
                "model": self.model_id,
                **prompt,
                **self.generation_args}
            
async def send_request(request):
    headers = {
        "Accept": "application/json" if request.mode == "sync" else "text/event-stream"}

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            if request.mode == "sync":
                response = await client.post(request.endpoint, json=request.payload, headers=headers)
                response.raise_for_status()
                yield response.json()

            elif request.mode == "stream":
                async with client.stream("POST", request.endpoint, json=request.payload, headers=headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data"):
                            yield line[6:]

        except httpx.HTTPStatusError as e:
            yield {"error": f"HTTP error {e.response.status_code}", "detail": str(e)}

        except httpx.RequestError as e:
            yield {"error": "Request failed", "detail": str(e)}

        except Exception as e:
            yield {"error": "Unhandled exception", "detail": str(e)}

async def sync_(request):
    async for response in send_request(request):
        print(response)
        print()

async def stream(request):
    async for data in send_request(request):
        print(data)
        print()

def run_throughput_benchmark(model, task, mode, repeat, parallel, output_path):
    pass

def run_latency_benchmark(model, task, mode, repeat, output_path):
    for prompt in prompts(model, task):
        request = CompletionRequest(model, task, prompt, mode)

        if mode == "sync":
            asyncio.run(sync_(request))
        elif mode == "stream":
            asyncio.run(stream(request))

def main():
    args = parse_args()

    output_filename = f"{args.model}-{args.task}-{args.mode}-r{args.repeat}"
    if args.throughput_only:
        output_filename += f"-p{args.parallel}"
    output_filename += ".csv"

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    if args.throughput_only:
        run_throughput_benchmark(
            model=args.model,
            task=args.task,
            mode=args.mode,
            repeat=args.repeat,
            parallel=args.parallel,
            output_path=output_path)
    else:
        run_latency_benchmark(
            model=args.model,
            task=args.task,
            mode=args.mode,
            repeat=args.repeat,
            output_path=output_path)

if __name__ == "__main__":
    main()
