import argparse
import sys
import os
import torch
import logging

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

def profile_forward_pass(model, input_ids, device, num_iterations=10):
    """Profile forward pass only (inference)"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Profiled iterations
    with torch.no_grad():
        for i in range(num_iterations):
            torch.cuda.nvtx.range_push(f"Forward_Pass_Iteration_{i}")
            _ = model(input_ids)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

def profile_forward_backward(model, input_ids, device, num_iterations=10):
    """Profile forward + backward pass (training without optimizer)"""
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    
    # Warmup
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Profiled iterations
    for i in range(num_iterations):
        torch.cuda.nvtx.range_push(f"Training_Step_{i}")
        
        torch.cuda.nvtx.range_push(f"Forward")
        model.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = logits.sum()
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push(f"Backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

def profile_full_training_step(model, input_ids, device, learning_rate=1e-4, num_iterations=10):
    """Profile forward + backward + optimizer step"""
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Warmup
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        optimizer.step()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Profiled iterations
    for i in range(num_iterations):
        torch.cuda.nvtx.range_push(f"Full_Training_Step_{i}")
        
        torch.cuda.nvtx.range_push(f"Forward")
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = logits.sum()
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push(f"Backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push(f"Optimizer_Step")
        optimizer.step()
        torch.cuda.nvtx.range_pop()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

def main():
    parser = argparse.ArgumentParser(description="Profile Transformer model with nsys")
    parser.add_argument("--model_size", type=str, default="small", 
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
                       choices=["forward", "forward_backward", "full_training"],
                       help="Profiling mode")
    parser.add_argument("--num_iterations", type=int, default=10,
                       help="Number of iterations to profile")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type != 'cuda':
        logger.error("CUDA device required for nsys profiling")
        sys.exit(1)
    
    logger.info(f"Profiling {args.model_size} model with context_length={args.context_length}")
    logger.info(f"Mode: {args.mode}")
    
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
    
    logger.info("Starting profiling...")
    
    # Profile based on mode
    if args.mode == "forward":
        profile_forward_pass(model, input_ids, device, args.num_iterations)
    elif args.mode == "forward_backward":
        profile_forward_backward(model, input_ids, device, args.num_iterations)
    elif args.mode == "full_training":
        profile_full_training_step(model, input_ids, device, args.num_iterations)
    
    logger.info("Profiling complete!")

if __name__ == "__main__":
    main()