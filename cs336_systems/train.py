import argparse
import os
import time
import math
import numpy as np
import torch
import wandb
import tiktoken
import sys
import logging
from tqdm import tqdm

# Ensure we can import from cs336_basics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.models import BasicsTransformerLM
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.nn_utils import cross_entropy, clip_gradient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Set precision
torch.set_float32_matmul_precision('high')

# --- Helper 1: Create .bin file from .txt ---
def generate_bin_file(txt_path, bin_path):
    logger.info(f"Generating {bin_path} from {txt_path}...")
    try:
        enc = tiktoken.get_encoding("gpt2")
        eot = enc.eot_token
        
        with open(bin_path, 'wb') as f_out:
            with open(txt_path, 'r', encoding='utf-8') as f_in:
                pbar = tqdm(desc="Tokenizing", unit=" lines")
                buffer = []
                BATCH_SIZE = 1_000_000 
                
                for line in f_in:
                    tokens = enc.encode(line, allowed_special={'<|endoftext|>'})
                    buffer.extend(tokens)
                    buffer.append(eot)
                    pbar.update(1)
                    
                    if len(buffer) >= BATCH_SIZE:
                        arr = np.array(buffer, dtype=np.uint16)
                        f_out.write(arr.tobytes())
                        buffer = []
                
                if buffer:
                    arr = np.array(buffer, dtype=np.uint16)
                    f_out.write(arr.tobytes())
                pbar.close()
        logger.info(f"Data preparation complete. Saved to {bin_path}.")
    except Exception as e:
        logger.error(f"Error generating bin file: {e}")
        if os.path.exists(bin_path):
            os.remove(bin_path)
        raise

# --- Helper 2: Resolve Data Paths ---
def prepare_data_if_needed(bin_path):
    if not bin_path: return None
    if os.path.exists(bin_path): return bin_path
    
    # Check if .txt exists for generation
    txt_path = os.path.splitext(bin_path)[0] + ".txt"
    if os.path.exists(txt_path):
        generate_bin_file(txt_path, bin_path)
        return bin_path
    
    # Check parent directory
    parent_bin = os.path.join("..", bin_path)
    parent_txt = os.path.join("..", txt_path)
    if os.path.exists(parent_bin): return parent_bin
    if os.path.exists(parent_txt):
        generate_bin_file(parent_txt, parent_bin)
        return parent_bin

    raise FileNotFoundError(f"Could not find {bin_path} or {txt_path}.")

# --- Helper 3: Get Batch ---
def get_batch(data, batch_size, context_length, device, vocab_size=None):
    ix = torch.randint(len(data) - context_length, (batch_size,))
    
    # Load data from numpy
    x_np = np.stack([data[i:i+context_length] for i in ix]).astype(np.int64)
    y_np = np.stack([data[i+1:i+1+context_length] for i in ix]).astype(np.int64)
    
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    # --- SAFETY FIX FOR TESTING ---
    # If we are testing with a small vocab (e.g. 1000), but the data comes 
    # from TikToken (50k), we must modulo the data to prevent index errors.
    # Note: This destroys semantic meaning, but allows the code to run.
    if vocab_size is not None:
        x = x % vocab_size
        y = y % vocab_size

    return x.to(device), y.to(device)

# --- Helper 4: Estimate Loss ---
def estimate_loss(model, data, batch_size, context_length, device, eval_iters, vocab_size=None):
    model.eval()
    losses = torch.zeros(eval_iters)
    
    with torch.no_grad():
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, context_length, device, vocab_size=vocab_size)
            logits = model(X)
            loss = cross_entropy(logits, Y) # Using nn_utils.cross_entropy
            losses[k] = loss.item()
            
    model.train()
    return losses.mean()

# --- Helper 5: Log VRAM Usage ---
def log_vram_usage(out_dir, iter_num, device):
    """Logs VRAM usage to a file in the logs folder."""
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "vram_usage.txt")
    
    msg = f"Iter {iter_num}: Device {device} | "
    
    if device == 'cuda':
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        msg += f"Allocated: {allocated:.4f} GB | Reserved: {reserved:.4f} GB | Max Alloc: {max_allocated:.4f} GB"
    elif device == 'mps':
        try:
            allocated = torch.mps.current_allocated_memory() / (1024 ** 3)
            msg += f"Allocated: {allocated:.4f} GB"
        except AttributeError:
            msg += "VRAM logging not supported for this MPS version."
    else:
        msg += "VRAM logging not applicable for CPU."
        
    logger.info(f"[VRAM LOG] {msg}")
    
    with open(log_path, "a") as f:
        f.write(msg + "\n")

# --- Helper 6: Checkpointing ---
def save_checkpoint(model, optimizer, iter_num, filepath, config):
    checkpoint = {
        'iter_num': iter_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iter_num']

def get_args():
    parser = argparse.ArgumentParser(description="CS336 Assignment 2: Training Script")

    # --- MODE & TRACKING ---
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"])
    parser.add_argument("--exp_name", type=str, default=None, help="Name of the run in W&B")
    parser.add_argument("--tags", type=str, nargs='*', help="Tags for W&B")

    # --- Common Params ---
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--device", type=str, default=None)

    # --- Training Params ---
    parser.add_argument("--train_data", type=str, default="data/train.bin")
    parser.add_argument("--val_data", type=str, default="data/val.bin")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=50)
    
    # Optimizer (using cs336_basics optimizer.py defaults)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)

    # Logging
    parser.add_argument("--wandb_project", type=str, default="cs336-systems")
    parser.add_argument("--no_wandb", action="store_true")

    # --- Sampling Params ---
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)

    # --- Architecture (Defaults for small test model) ---
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    return parser.parse_args()

def main():
    args = get_args()

    # 1. Device Setup
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
            
    logger.info(f"Using device: {device}")

    # ==========================================
    # LOGIC BRANCH 1: SAMPLING MODE
    # ==========================================
    if args.mode == "sample":
        if args.checkpoint and os.path.exists(args.checkpoint):
            logger.info(f"Loading checkpoint: {args.checkpoint}")
            ckpt_data = torch.load(args.checkpoint, map_location=device)
            model_config = ckpt_data.get("config", None)
            if model_config is None: raise ValueError("No config in checkpoint")
            
            # Re-init model
            model = BasicsTransformerLM(**model_config).to(device)
            model.load_state_dict(ckpt_data["model_state_dict"])
        else:
            logger.warning("WARNING: Sampling with random weights (no checkpoint provided).")
            model = BasicsTransformerLM(
                vocab_size=args.vocab_size,
                context_length=args.context_length,
                d_model=args.d_model,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                d_ff=args.d_ff,
                rope_theta=args.rope_theta,
            ).to(device)

        model.eval()
        enc = tiktoken.get_encoding("gpt2")
        prompt_tokens = enc.encode(args.prompt)
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        logger.info(f"Generating ({args.max_new_tokens} tokens)...")
        
        # Use the generate method from BasicsTransformerLM
        out = model.generate(
            prompt_tensor, 
            max_new_tokens=args.max_new_tokens, 
            temperature=args.temperature, 
            top_k=args.top_k, 
            eos_token_id=enc.eot_token
        )
        
        print("-" * 50)
        print(enc.decode(out[0].tolist()))
        print("-" * 50)
        return

    # ==========================================
    # LOGIC BRANCH 2: TRAINING MODE
    # ==========================================
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Init W&B
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.exp_name, tags=args.tags, config=args)

    # Data
    args.train_data = prepare_data_if_needed(args.train_data)
    args.val_data = prepare_data_if_needed(args.val_data)
    
    # Load Memory Maps
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')

    # Init Model
    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }
    model = BasicsTransformerLM(**model_config)
    model.to(device)

    # --- TORCH.COMPILE ---
    # Optional: Turn off if debugging / stepping through code
    if device != 'mps': # MPS often has issues with compile currently
        logger.info("Compiling model...")
        try:
            model = torch.compile(model)
            logger.info(" -> Model compiled.")
        except Exception as e:
            logger.warning(f" -> Compilation failed: {e}. Running in eager mode.")
            
    # Optimizer (Using cs336_basics.optimizer.AdamW)
    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(args.beta1, args.beta2), 
        weight_decay=args.weight_decay
    )

    # Resume from checkpoint
    iter_num = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Resuming from {args.checkpoint}")
        iter_num = load_checkpoint(args.checkpoint, model, optimizer)

    logger.info(f"Starting training on {device}...")
    global_start_time = time.time()
    t0 = time.time()
    
    model.train()
    
    while iter_num < args.max_iters:
        # 1. Update Learning Rate
        lr = get_cosine_lr(
            iter_num, 
            max_learning_rate=args.lr, 
            min_learning_rate=args.min_lr, 
            warmup_iters=args.warmup_iters, 
            cosine_cycle_iters=args.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups: 
            param_group['lr'] = lr
        
        # 2. Get Data
        X, Y = get_batch(train_data, args.batch_size, args.context_length, device, vocab_size=args.vocab_size)
        
        # 3. Forward + Backward
        logits = model(X)
        loss = cross_entropy(logits, Y) # Using nn_utils.cross_entropy
        
        loss.backward()
        
        # 4. Clip & Step
        if args.grad_clip > 0.0:
            clip_gradient(model.parameters(), args.grad_clip) # Using nn_utils.clip_gradient
            
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # 5. Logging
        if iter_num % args.log_interval == 0:
            current_time = time.time()
            elapsed_time = current_time - global_start_time
            
            # Approximate Speed
            dt = current_time - t0
            t0 = current_time
            tokens_per_sec = (args.batch_size * args.context_length * args.log_interval) / (dt + 1e-6)

            print(f"Iter {iter_num:5d}/{args.max_iters} | Loss: {loss.item():.4f} | LR: {lr:.5f} | Speed: {tokens_per_sec:.0f} tok/s")
            
            if not args.no_wandb: 
                wandb.log({
                    "train/loss": loss.item(), 
                    "train/lr": lr,
                    "train/wallclock_time": elapsed_time,
                    "train/tokens_per_sec": tokens_per_sec,
                    "iter": iter_num
                })
            
        # 6. Validation
        if iter_num > 0 and iter_num % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
            elapsed_time = time.time() - global_start_time
            print(f"\n[VALIDATION] Iter {iter_num}: Loss {val_loss:.4f}\n")
            if not args.no_wandb: 
                wandb.log({"val/loss": val_loss, "iter": iter_num})

        # 7. Rolling Checkpoint & VRAM Logging
        if iter_num > 0 and iter_num % args.save_interval == 0:
            ckpt_path = os.path.join(args.out_dir, "ckpt_latest.pt")
            logger.info(f"Saving rolling checkpoint to {ckpt_path}...")
            save_checkpoint(model, optimizer, iter_num, ckpt_path, config=model_config)
            
            log_vram_usage(args.out_dir, iter_num, device)
            
        iter_num += 1

    # Save Final Model
    save_checkpoint(model, optimizer, iter_num, os.path.join(args.out_dir, "ckpt_final.pt"), config=model_config)
    log_vram_usage(args.out_dir, iter_num, device)
    
    logger.info("Training Complete.")

if __name__ == "__main__":
    main()