import argparse
import sys
import os
import torch
import logging

# Ensure we can import from cs336_basics and local utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.model import BasicsTransformerLM
import benchmark_utils as utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_suite(device):
    """
    Runs the full benchmark suite for all model sizes defined in the assignment
    and prints the table (Part b).
    """
    logger.info(f"Starting Benchmark Suite on {device}...")
    
    # 1. Get all model specifications
    all_specs = utils.get_model_specs()
    
    # 2. Filter for CPU (Small, Medium, Large only) to prevent stalling
    #    We create a new dictionary with only the keys we can handle right now.
    specs = {k: all_specs[k] for k in ["small", "medium"]} # , "large"

    # --- WHEN YOU HAVE A GPU, UNCOMMENT THE LINE BELOW AND REMOVE THE FILTER ABOVE ---
    # specs = all_specs 
    # -------------------------------------------------------------------------------

    results = []

    # Constants for the suite
    VOCAB_SIZE = 10000
    BATCH_SIZE = 4
    CONTEXT_LENGTH = 128
    
    for size_name, config in specs.items():
        logger.info(f"Benchmarking {size_name.upper()}...")
        
        # Initialize Model
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

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM for {size_name}")
                results.append({"Size": size_name, "Forward (ms)": "OOM", "Backward (ms)": "OOM"})
                torch.cuda.empty_cache()
                continue
            raise e

        # Generate Batch
        input_ids = utils.generate_batch(VOCAB_SIZE, BATCH_SIZE, CONTEXT_LENGTH, device)

        # Forward
        fwd_avg, fwd_std = utils.benchmark_pass(model, input_ids, "forward", warmup_steps=5, num_steps=10)
        
        # Backward
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

        # Cleanup
        del model, input_ids
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Use utility to print the table
    utils.print_results_table(results)

def main():
    parser = argparse.ArgumentParser(description="CS336 Assignment 2 Benchmark")
    
    # Mode selection
    parser.add_argument("--suite", action="store_true", help="Run the full table generation (Part b)")
    
    # Arguments for Single Run (Part a)
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
    if device.type == 'cpu':
        logger.warning("WARNING: Running on CPU. Results will be slow.")

    # --- EXECUTION BRANCH ---
    if args.suite:
        # Run Part (b): The Table (Filtered for CPU)
        run_suite(device)
    else:
        # Run Part (a): The Single Deliverable
        logger.info(f"Running Single Benchmark ({args.mode}) on {device}...")
        
        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=10000.0,
        ).to(device)

        input_ids = utils.generate_batch(args.vocab_size, args.batch_size, args.context_length, device)
        
        avg, std = utils.benchmark_pass(
            model, input_ids, args.mode, 
            warmup_steps=args.warmup_steps, 
            num_steps=args.num_steps
        )
        
        print("-" * 40)
        print(f"Mode: {args.mode.upper()}")
        print(f"Time: {avg:.4f} ms ± {std:.4f} ms")
        print("-" * 40)

if __name__ == "__main__":
    main()