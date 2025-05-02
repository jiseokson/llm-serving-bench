import os, sys
import argparse

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

def run_throughput_benchmark(model, task, mode, repeat, parallel, output_path):
    print("run_throughput_benchmark")
    print(model, task, mode, repeat, parallel, output_path)

def run_latency_benchmark(model, task, mode, repeat, output_path):
    print("run_latency_benchmark")
    print(model, task, mode, repeat, output_path)

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
            output_path=output_path
        )
    else:
        run_latency_benchmark(
            model=args.model,
            task=args.task,
            mode=args.mode,
            repeat=args.repeat,
            output_path=output_path
        )

if __name__ == "__main__":
    main()
