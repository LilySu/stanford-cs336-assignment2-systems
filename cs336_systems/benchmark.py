import argparse
import sys
import os
import torch
import logging
import timeit
import statistics

# -----------------------------------------------------------------------------
# Path Setup
# -----------------------------------------------------------------------------
# Ensure we can import from the current directory and the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# 1. Import Model
try:
    from cs336_basics.model import BasicsTransformerLM
except ImportError:
    try:
        from cs336_basics.models import BasicsTransformerLM
    except ImportError:
        print("Error: Could not import BasicsTransformerLM. Check your folder structure.")
        sys.exit(1)

# 2. Import Utils (Required for Suite)
# We try to import benchmark_utils. If it fails, suite mode will crash, but single mode works.
try:
    import benchmark_utils as utils
except ImportError:
    utils = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# PART A: Single Run Benchmark (With NVTX for Profiling)
# -----------------------------------------------------------------------------
def run_single(args, device):
    logger.info(f"Initializing model on {device}...")
    
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000.0,
    ).to(device)

    input_ids = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length), 
        device=device, dtype=torch.int64
    )

    if args.mode == "backward":
        model.train()
        for param in model.parameters():
            param.requires_grad = True
    else:
        model.eval()

    def run_forward():
        _ = model(input_ids)

    def run_forward_backward():
        model.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = logits.sum() 
        loss.backward()

    step_fn = run_forward_backward if args.mode == "backward" else run_forward

    # Warmup
    for _ in range(args.warmup_steps):
        step_fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timing
    timings = []
    timer = timeit.default_timer

    for i in range(args.num_steps):
        if device.type == 'cuda':
            torch.cuda.nvtx.range_push(f"Step {i}")
            torch.cuda.synchronize()
            
        start_time = timer()
        step_fn()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            
        end_time = timer()
        timings.append(end_time - start_time)

    avg_time = statistics.mean(timings)
    std_dev = statistics.stdev(timings) if len(timings) > 1 else 0.0

    print("-" * 40)
    print(f"Mode: {args.mode.upper()}")
    print(f"Time: {avg_time*1000:.4f} ms ± {std_dev*1000:.4f} ms")
    print("-" * 40)

# -----------------------------------------------------------------------------
# PART B: Full Suite (Uses benchmark_utils)
# -----------------------------------------------------------------------------
def run_suite(device):
    if utils is None:
        logger.error("CRITICAL ERROR: 'benchmark_utils.py' not found.")
        logger.error("You cannot run --suite without benchmark_utils.py in this folder.")
        sys.exit(1)

    logger.info(f"Starting Benchmark Suite on {device}...")
    
    all_specs = utils.get_model_specs()
    
    # Auto-select models based on device
    if device.type == 'cuda':
        specs = all_specs # Run ALL models on GPU
    else:
        logger.warning("Running on CPU: Restricting to 'small' and 'medium' only.")
        specs = {k: all_specs[k] for k in ["small", "medium"] if k in all_specs}

    results = []
    VOCAB_SIZE = 10000
    BATCH_SIZE = 4
    CONTEXT_LENGTH = 128
    
    for size_name, config in specs.items():
        logger.info(f"Benchmarking {size_name.upper()}...")
        
        try:
            model = BasicsTransformerLM(
                vocab_size=VOCAB_SIZE,
                context_length=CONTEXT_LENGTH,
                d_model=config["d_model"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                d_ff=config["d_ff"],
                rope_theta=10000.0,
            ).to(device)

            input_ids = utils.generate_batch(VOCAB_SIZE, BATCH_SIZE, CONTEXT_LENGTH, device)

            # Measure Forward
            fwd_avg, fwd_std = utils.benchmark_pass(model, input_ids, "forward", warmup_steps=5, num_steps=10)
            
            # Measure Backward
            bwd_avg, bwd_std = utils.benchmark_pass(model, input_ids, "backward", warmup_steps=5, num_steps=10)
            
            results.append({
                "Size": size_name,
                "d_model": config["d_model"],
                "d_ff": config["d_ff"],
                "num_layers": config["num_layers"],
                "num_heads": config["num_heads"],
                "Forward (ms)": f"{fwd_avg:.2f} ± {fwd_std:.2f}",
                "Backward (ms)": f"{bwd_avg:.2f} ± {bwd_std:.2f}"
            })

            # Cleanup to prevent OOM
            del model, input_ids
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM for {size_name}")
                results.append({"Size": size_name, "Forward (ms)": "OOM", "Backward (ms)": "OOM"})
                torch.cuda.empty_cache()
            else:
                raise e

    utils.print_results_table(results)

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", action="store_true", help="Run the full table generation (Part b)")
    
    # Arguments for Single Run
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "backward"])
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=10)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.suite:
        run_suite(device)
    else:
        run_single(args, device)

if __name__ == "__main__":
    main()