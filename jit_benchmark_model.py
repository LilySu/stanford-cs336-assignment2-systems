import torch
import time
import pandas as pd
import sys
import os
import datetime
from cs336_systems.cs336_basics.model import BasicsTransformerLM

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_SPECS = {
    "small": {"d_model": 512, "num_layers": 6, "num_heads": 8, "d_ff": 2048},
    "medium": {"d_model": 1024, "num_layers": 12, "num_heads": 16, "d_ff": 4096},
}

BATCH_SIZE = 4
CONTEXT_LENGTH = 128
VOCAB_SIZE = 10000
NUM_ITER = 20
WARMUP = 5

# Generate Timestamped Filename
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE = f"{timestamp}_model_jit_results.csv"

def benchmark_model_run(model, input_ids, mode="forward"):
    """Benchmarks a model for a specific mode."""
    
    optimizer = None
    if mode == "training":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()
    else:
        model.eval()

    # Step Function
    def step():
        if mode == "forward":
            with torch.no_grad():
                _ = model(input_ids)
        elif mode == "training":
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = logits.sum()
            loss.backward()
            optimizer.step()

    # 1. Warmup
    # Critical for JIT to compile the graph
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()

    # 2. Timing
    start_time = time.time()
    for _ in range(NUM_ITER):
        step()
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    avg_time_ms = (total_time / NUM_ITER) * 1000
    return avg_time_ms

def run_suite():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Full Model JIT Benchmark on {device}...")
    print(f"Results will be saved to: {OUTPUT_FILE}")
    
    results = []

    for size_name, config in MODEL_SPECS.items():
        print(f"Benchmarking {size_name.upper()} model...")
        
        # Init Model
        model = BasicsTransformerLM(
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=10000.0,
        ).to(device)
        
        input_ids = torch.randint(
            0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), 
            device=device, dtype=torch.int64
        )
        
        # --- 1. Vanilla Benchmark ---
        vanilla_fwd = benchmark_model_run(model, input_ids, mode="forward")
        vanilla_train = benchmark_model_run(model, input_ids, mode="training")
        
        # --- 2. Compiled Benchmark ---
        print(f"  Compiling model...")
        # Compile the model
        compiled_model = torch.compile(model)
        
        jit_fwd = benchmark_model_run(compiled_model, input_ids, mode="forward")
        jit_train = benchmark_model_run(compiled_model, input_ids, mode="training")
        
        results.append({
            "Model Size": size_name,
            "Vanilla Fwd (ms)": f"{vanilla_fwd:.2f}",
            "Compiled Fwd (ms)": f"{jit_fwd:.2f}",
            "Speedup Fwd": f"{vanilla_fwd/jit_fwd:.2f}x",
            "Vanilla Train (ms)": f"{vanilla_train:.2f}",
            "Compiled Train (ms)": f"{jit_train:.2f}",
            "Speedup Train": f"{vanilla_train/jit_train:.2f}x"
        })
        
        del model, compiled_model, input_ids
        torch.cuda.empty_cache()

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print("\nBenchmark Results:")
    print(df.to_markdown(index=False))
    print(f"\nSaved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    run_suite()