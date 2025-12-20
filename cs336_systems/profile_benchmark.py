import argparse
import sys
import os
import torch
import logging
from contextlib import nullcontext

# -----------------------------------------------------------------------------
# Path Setup
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from cs336_basics.model import BasicsTransformerLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Model configurations from Table 1
MODEL_SPECS = {
    "small": {"d_model": 512, "num_layers": 6, "num_heads": 8, "d_ff": 2048},
    "medium": {"d_model": 1024, "num_layers": 12, "num_heads": 16, "d_ff": 4096},
    "large": {"d_model": 2048, "num_layers": 24, "num_heads": 32, "d_ff": 8192},
}

def get_autocast_context(device, enable_mixed_precision):
    """Returns the autocast context for BF16 if enabled."""
    if enable_mixed_precision and device.type == 'cuda':
        return torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()

def benchmark_run(model, input_ids, device, mixed_precision, mode, num_iterations, profile_memory=False):
    """
    Unified runner for profiling compute, timing, and memory.
    Handles NVTX markers for Nsight Systems and CUDA Events for console timing.
    """
    
    # 1. Setup Model State
    if mode == "forward":
        model.eval()
    else:
        model.train()
        for param in model.parameters(): 
            param.requires_grad = True

    # 2. Setup Context & Optimizer
    ctx = get_autocast_context(device, mixed_precision)
    optimizer = None
    if mode == "full_training":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # --- Helper: Defines a single step with NVTX tagging ---
    def run_step(step_idx):
        # Outer NVTX Range
        range_name = f"{mode}_step_{step_idx}"
        torch.cuda.nvtx.range_push(range_name)
        
        # A. Forward
        torch.cuda.nvtx.range_push("Forward")
        if mode == "full_training": 
            optimizer.zero_grad(set_to_none=True)
        elif mode == "forward_backward": 
            model.zero_grad(set_to_none=True)
            
        with ctx:
            logits = model(input_ids)
            if mode != "forward":
                loss = logits.sum()
        torch.cuda.nvtx.range_pop() # End Forward

        # B. Backward
        if mode != "forward":
            torch.cuda.nvtx.range_push("Backward")
            loss.backward()
            torch.cuda.nvtx.range_pop() # End Backward

        # C. Optimizer
        if mode == "full_training":
            torch.cuda.nvtx.range_push("Optimizer_Step")
            optimizer.step()
            torch.cuda.nvtx.range_pop() # End Optimizer

        # Sync within the step (optional, but good for nsys visualization boundaries)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        torch.cuda.nvtx.range_pop() # End Outer Range

    # 3. Warmup (Run 3 times to stabilize clock/caches)
    for i in range(3):
        run_step(i)
    
    # =======================================================
    # LOGIC BRANCH 1: MEMORY PROFILING (Part D)
    # =======================================================
    if profile_memory:
        logger.info("Starting PyTorch Memory Recording...")
        
        try:
            # Start recording
            torch.cuda.memory._record_memory_history(max_entries=1000000)
            
            # Run one step to capture the memory pattern
            run_step(999)
            
            # Save snapshot
            snapshot_filename = "memory_snapshot.pickle"
            logger.info(f"Dumping memory snapshot to {snapshot_filename}...")
            torch.cuda.memory._dump_snapshot(snapshot_filename)
            
            # Stop recording
            torch.cuda.memory._record_memory_history(enabled=None)
            logger.info("Memory profiling complete.")
        except AttributeError:
            logger.error("Error: PyTorch 2.1+ required for _record_memory_history.")
        
        return 0.0 # Return 0 ms since we aren't timing

    # =======================================================
    # LOGIC BRANCH 2: TIMING & NSIGHT PROFILING (Part C)
    # =======================================================
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Start Timing
    start_event.record()
    
    # Run the Loop
    for i in range(num_iterations):
        run_step(i)
        
    end_event.record()
    torch.cuda.synchronize()
    
    # Calculate Average
    total_time_ms = start_event.elapsed_time(end_event)
    return total_time_ms / num_iterations

def main():
    parser = argparse.ArgumentParser(description="CS336 Profiling Script")
    parser.add_argument("--model_size", type=str, default="small", 
                        choices=["small", "medium", "large"], help="Model size config")
    parser.add_argument("--context_length", type=int, default=128, help="Context length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--mode", type=str, default="forward_backward", 
                        choices=["forward", "forward_backward", "full_training"], 
                        help="Operation mode to profile")
    parser.add_argument("--num_iterations", type=int, default=10, help="Iterations to average over")
    
    # --- New Flags ---
    parser.add_argument("--mixed_precision", action="store_true", help="Enable BF16 Mixed Precision (Part C)")
    parser.add_argument("--profile_memory", action="store_true", help="Generate memory_snapshot.pickle (Part D)")
    
    args = parser.parse_args()
    
    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        logger.warning("WARNING: CUDA not available. Profiling results will be invalid.")
    
    # Init Model
    config = MODEL_SPECS[args.model_size]
    logger.info(f"Initializing {args.model_size} model on {device}...")
    
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,
    ).to(device)
    
    # Init Data
    input_ids = torch.randint(
        0, args.vocab_size, 
        (args.batch_size, args.context_length), 
        device=device, dtype=torch.int64
    )
    
    logger.info(f"Running Benchmark | Mode: {args.mode} | Mixed Precision: {args.mixed_precision}")
    
    # Run Benchmark
    avg_time = benchmark_run(
        model, input_ids, device, 
        args.mixed_precision, args.mode, 
        args.num_iterations, args.profile_memory
    )
    
    # Report Results
    if not args.profile_memory:
        precision_tag = "BF16" if args.mixed_precision else "FP32"
        print(f"\nRESULT | {args.model_size} | {args.mode} | {precision_tag} | {avg_time:.4f} ms/iter\n")

if __name__ == "__main__":
    main()