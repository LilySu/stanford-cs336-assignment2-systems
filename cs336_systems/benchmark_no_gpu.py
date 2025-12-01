import timeit
import torch
import argparse
import statistics
import logging
import sys
import os

# Add the parent directory to sys.path to allow importing cs336_basics 
# if running directly from the folder, though running from root is preferred.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.models import BasicsTransformerLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def benchmark(args):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        logger.warning("WARNING: Benchmarking on CPU. Results will be slow and not representative of GPU training.")

    # 2. Initialize Model
    logger.info(f"Initializing model with d_model={args.d_model}, layers={args.num_layers}...")
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000.0, # Default per your provided code
    ).to(device)

    # 3. Generate Random Batch
    # Input shape: (batch_size, context_length)
    input_ids = torch.randint(
        0, 
        args.vocab_size, 
        (args.batch_size, args.context_length), 
        device=device,
        dtype=torch.int64 # Expecting Int/Long for embeddings
    )

    # Gradient handling
    if args.mode == "backward":
        model.train()
        # We don't explicitly need input gradients, but we need parameter gradients
        for param in model.parameters():
            param.requires_grad = True
    else:
        model.eval()

    # Define the step function based on mode
    def run_forward():
        _ = model(input_ids)

    def run_forward_backward():
        # Clear previous gradients
        model.zero_grad(set_to_none=True)
        # Forward
        logits = model(input_ids)
        # Create a dummy loss (scalar) to backpropagate
        # We sum output to get a scalar, then call backward
        loss = logits.sum() 
        loss.backward()

    # Select the function to benchmark
    step_fn = run_forward_backward if args.mode == "backward" else run_forward

    # 4. Warmup
    logger.info(f"Running {args.warmup_steps} warmup steps ({args.mode})...")
    for _ in range(args.warmup_steps):
        step_fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # 5. Timing
    logger.info(f"Measuring {args.num_steps} steps...")
    timings = []
    
    # Use timeit.default_timer for high resolution
    timer = timeit.default_timer

    for i in range(args.num_steps):
        # Synchronize before starting timer to ensure previous GPU ops are done
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        start_time = timer()
        
        step_fn()
        
        # Synchronize after step to ensure we time the actual GPU execution
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        end_time = timer()
        timings.append(end_time - start_time)

    # 6. Statistics
    avg_time = statistics.mean(timings)
    std_dev = statistics.stdev(timings) if len(timings) > 1 else 0.0

    print("-" * 40)
    print(f"Results for Mode: {args.mode.upper()}")
    print(f"Configuration: Batch={args.batch_size}, SeqLen={args.context_length}, d_model={args.d_model}, Layers={args.num_layers}")
    print(f"Average Time: {avg_time*1000:.4f} ms")
    print(f"Std Dev:      {std_dev*1000:.4f} ms")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM Forward/Backward Passes")

    # Hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    
    # Benchmark settings
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mode", type=str, choices=["forward", "backward"], default="forward", 
                        help="Measure 'forward' pass only, or 'backward' (which includes forward + backward)")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=10)

    args = parser.parse_args()
    
    # Validation
    if args.d_model % args.num_heads != 0:
        raise ValueError(f"d_model ({args.d_model}) must be divisible by num_heads ({args.num_heads})")

    benchmark(args)