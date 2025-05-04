import os
import sys
import json
import time
import asyncio
import argparse

import httpx
import pandas as pd

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
            
class Logger:
    def __init__(self, output_path):
        self.output_path = output_path

        self.data = {}
        self.column_counter = {}

    def add_column(self, column, data, exist_ok=False):
        if exist_ok and self.column_counter.get(column, 0) > 0:
            return

        column_count = self.column_counter.get(column, 0) + 1
        self.column_counter[column] = column_count

        self.data[column if exist_ok else f"{column}#{column_count}"] = data

    def add_row(self, column, data):
        datas = self.data.get(column, [])
        datas.append(data)
        self.data[column] = datas

    def checkout(self):
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path, index=False)

async def send_request(request):
    headers = {
        "Accept": "application/json" if request.mode == "sync" else "text/event-stream"}

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            if request.mode == "sync":
                start = time.perf_counter()
                response = await client.post(request.endpoint, json=request.payload, headers=headers)
                end = time.perf_counter()

                response.raise_for_status()

                yield response.json(), end - start

            elif request.mode == "stream":
                start = time.perf_counter()

                async with client.stream("POST", request.endpoint, json=request.payload, headers=headers) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        try:
                            yield json.loads(line[6:] if line.startswith("data") else line), time.perf_counter() - start
                        except:
                            continue

        except httpx.HTTPStatusError as e:
            yield {"error": f"HTTP error {e.response.status_code}", "detail": str(e)}, None

        except httpx.RequestError as e:
            yield {"error": "Request failed", "detail": str(e)}, None

        except Exception as e:
            yield {"error": "Unhandled exception", "detail": str(e)}, None

async def evaluate_sync_latency(model, task, mode, logger):
    prompt_tokens = []
    completion_tokens = []
    total_latencies = []
    token_generation_speeds = []

    for i, prompt in enumerate(prompts(model, task)):
        request = CompletionRequest(model, task, prompt, mode)

        async for response, latency in send_request(request):
            prompt_tokens.append(response["usage"]["prompt_tokens"] if latency is not None else None)
            completion_tokens.append(response["usage"]["completion_tokens"] if latency is not None else None)
            total_latencies.append(latency)
            token_generation_speeds.append(completion_tokens[-1] / total_latencies[-1] if latency is not None else None)

            if latency is not None:
                print(
                    f"[Complete {i+1:3d}] {model2id[model]} {task} | "
                    f"Prompt tokens: {prompt_tokens[-1]} | "
                    f"Completion tokens: {completion_tokens[-1]} | "
                    f"Total Latency: {total_latencies[-1]:.3f} sec | "
                    f"Token Gen Speed: {token_generation_speeds[-1]:.3f} tok/s")
            else:
                print(f"[Error] {response}")

    logger.add_column("prompt_tokens", prompt_tokens, exist_ok=True)
    logger.add_column("completion_tokens", completion_tokens)
    logger.add_column("total_latency", total_latencies)
    logger.add_column("token_generation_speed", token_generation_speeds)

async def evaluate_stream_latency(model, task, mode, logger):
    completion_tokens = []
    p2fts = []
    total_latencies = []
    token_generation_speeds = []

    for i, prompt in enumerate(prompts(model, task)):
        token_latencies = []

        request, error = CompletionRequest(model, task, prompt, mode), None

        async for data, latency in send_request(request):
            token_latencies.append(latency)

            if latency is None:
                error = data

        if not error:
            completion_token = len(token_latencies)
            p2ft = token_latencies[0]
            total_latency = token_latencies[-1]
            token_generation_speed = len(token_latencies)/token_latencies[-1]

            print(
                f"[Complete {i+1:3d}] {model2id[model]} {task} | "
                f"Completion tokens: {completion_token} | "
                f"P2FT: {p2ft:.3f} sec | "
                f"Total Latency: {total_latency:.3f} sec | "
                f"Token Gen Speed: {token_generation_speed:.3f} tok/s")
        else:
            print(f"[Error] {error}")

        completion_tokens.append(completion_token if not error else None)
        p2fts.append(p2ft if not error else None)
        total_latencies.append(total_latency if not error else None)
        token_generation_speeds.append(token_generation_speed if not error else None)

    logger.add_column("completion_tokens", completion_tokens)
    logger.add_column("p2ft", p2fts)
    logger.add_column("total_latency", total_latencies)
    logger.add_column("token_generation_speed", token_generation_speeds)

async def evaluate_sync_throughput(model, task, mode, parallel, logger):
    async def worker(n):
        elapsed = 0
        requests = 0
        success_requests = 0
        completion_tokens = 0

        for i, prompt in enumerate(prompts(model, task)):
            request = CompletionRequest(model, task, prompt, mode)

            async for response, latency in send_request(request):
                elapsed += latency if latency is not None else 0
                completion_tokens += response["usage"]["completion_tokens"] if latency is not None else 0

                requests += 1

                if latency is not None:
                    print(
                        f"[Worker {n:2d} - Complete {i+1:3d}] {model2id[model]} {task} | "
                        f"Elapsed: {latency:.3f} sec | "
                        f"Completion tokens: {response["usage"]["completion_tokens"]}")

                    success_requests += 1

                else:
                    print(f"[Worker {n:2d} - Error] {response}")

        return [elapsed, requests, success_requests, completion_tokens]

    results = await asyncio.gather(*[worker(i + 1) for i in range(parallel)])

    all_elapses, all_requests, all_success_requests, all_completion_tokens = zip(*results)

    total_elapsed = max(elapsed for elapsed in all_elapses if elapsed is not None)
    total_request = sum(request for request in all_requests if request is not None)
    success_request = sum(success_request for success_request in all_success_requests if success_request is not None)
    completion_token = sum(completion_token for completion_token in all_completion_tokens if completion_token is not None)

    total_request_throughput = total_request / total_elapsed
    success_request_throughput = success_request / total_elapsed
    token_throughput = completion_token / total_elapsed

    logger.add_row("total_elapsed", total_elapsed)
    logger.add_row("total_request", total_request)
    logger.add_row("success_request", success_request)
    logger.add_row("completion_token", completion_token)
    logger.add_row("total_request_throughput", total_request_throughput)
    logger.add_row("success_request_throughput", success_request_throughput)
    logger.add_row("token_throughput", token_throughput)

    logger.checkout()

async def evaluate_stream_throughput(model, task, mode, parallel, logger):
    pass

def run_latency_benchmark(model, task, mode, repeat, output_path):
    logger = Logger(output_path)

    for i in range(repeat):
        if mode == "sync":
            asyncio.run(evaluate_sync_latency(model, task, mode, logger))
        elif mode == "stream":
            asyncio.run(evaluate_stream_latency(model, task, mode, logger))

    logger.checkout()

def run_throughput_benchmark(model, task, mode, repeat, parallel, output_path):
    logger = Logger(output_path)

    for i in range(repeat):
        if mode == "sync":
            asyncio.run(evaluate_sync_throughput(model, task, mode, parallel, logger))
        elif mode == "stream":
            asyncio.run(evaluate_stream_throughput(model, task, mode, parallel, logger))

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
