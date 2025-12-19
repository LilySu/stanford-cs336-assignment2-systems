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
    if enable_mixed_precision and device.type == 'cuda':
        return torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()

def profile_forward_pass(model, input_ids, device, mixed_precision, num_iterations=10):
    """Profile forward pass only (inference)"""
    model.eval()
    ctx = get_autocast_context(device, mixed_precision)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            with ctx:
                _ = model(input_ids)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Profiled iterations
    with torch.no_grad():
        for i in range(num_iterations):
            with ctx:
                _ = model(input_ids)
            if device.type == 'cuda':
                torch.cuda.synchronize()

def profile_full_training_step(model, input_ids, device, mixed_precision, learning_rate=1e-4, num_iterations=10):
    """Profile forward + backward + optimizer step"""
    model.train()
    ctx = get_autocast_context(device, mixed_precision)
    
    for param in model.parameters():
        param.requires_grad = True
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Warmup
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            logits = model(input_ids)
            loss = logits.sum()
        loss.backward()
        optimizer.step()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Profiled iterations
    for i in range(num_iterations):
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            logits = model(input_ids)
            loss = logits.sum()
        
        loss.backward()
        optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()

def main():
    parser = argparse.ArgumentParser(description="Profile Transformer model memory")
    parser.add_argument("--model_size", type=str, default="large", 
                       choices=["small", "medium", "large"],
                       help="Model size to profile")
    parser.add_argument("--context_length", type=int, default=128,
                       choices=[128, 256, 512, 1024],
                       help="Context length")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--vocab_size", type=int, default=10000,
                       help="Vocabulary size")
    parser.add_argument("--mode", type=str, default="forward",
                       choices=["forward", "full_training"],
                       help="Profiling mode")
    parser.add_argument("--num_iterations", type=int, default=5,
                       help="Number of iterations to profile")
    parser.add_argument("--mixed_precision", action="store_true",
                       help="Enable BF16 mixed precision")
    parser.add_argument("--profile_memory", action="store_true",
                       help="Enable PyTorch Memory Snapshot")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Profiling {args.model_size} model with context_length={args.context_length}")
    logger.info(f"Mode: {args.mode} | Mixed Precision: {args.mixed_precision} | Memory Profile: {args.profile_memory}")
    
    # Get model configuration
    config = MODEL_SPECS[args.model_size]
    
    # Create model
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,
    ).to(device)
    
    # Create input
    input_ids = torch.randint(
        0, args.vocab_size, 
        (args.batch_size, args.context_length), 
        device=device, 
        dtype=torch.int64
    )
    
    # Memory Profiling Context
    if args.profile_memory and device.type == 'cuda':
        logger.info("Enabling memory recording...")
        torch.cuda.memory._record_memory_history(max_entries=100000)

    logger.info("Starting run...")
    
    # Run Benchmark
    if args.mode == "forward":
        profile_forward_pass(model, input_ids, device, args.mixed_precision, args.num_iterations)
    elif args.mode == "full_training":
        profile_full_training_step(model, input_ids, device, args.mixed_precision, args.num_iterations)
    
    logger.info("Run complete!")

    # Dump Memory Snapshot
    if args.profile_memory and device.type == 'cuda':
        filename = f"memory_snapshot_{args.model_size}_{args.mode}_ctx{args.context_length}"
        if args.mixed_precision:
            filename += "_mp"
        filename += ".pickle"
        
        logger.info(f"Saving memory snapshot to {filename}...")
        try:
            torch.cuda.memory._dump_snapshot(filename)
        except Exception as e:
            logger.error(f"Failed to dump snapshot: {e}")
        
        torch.cuda.memory._record_memory_history(enabled=None)

if __name__ == "__main__":
    main()