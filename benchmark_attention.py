import torch
import time
import pandas as pd
import os
import sys

# Import your specific implementation
from cs336_systems.scaled_dot_product_attention import annotated_scaled_dot_product_attention

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BATCH_SIZE = 8
HEAD_DIMS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
NUM_ITER = 100
WARMUP = 10
OUTPUT_DIR = "assignment_results"
OUTPUT_FILE = "attention_benchmark.csv"

def format_memory(bytes_val):
    if bytes_val == "OOM": return "OOM"
    if bytes_val < 1024: return f"{bytes_val} B"
    elif bytes_val < 1024**2: return f"{bytes_val/1024:.2f} KB"
    elif bytes_val < 1024**3: return f"{bytes_val/1024**2:.2f} MB"
    else: return f"{bytes_val/1024**3:.2f} GB"

def run_benchmark():
    # 1. Setup Output Directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"===========================================================")
    print(f"Running Attention Benchmark on {device}")
    print(f"Saving records to: {csv_path}")
    print(f"===========================================================")
    
    results = []

    for d in HEAD_DIMS:
        for seq_len in SEQ_LENS:
            config_str = f"B={BATCH_SIZE}, Seq={seq_len}, D={d}"
            print(f"Benchmarking: {config_str:<30} |", end=" ")
            
            # Reset Memory for clean measurement
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            try:
                # (c) Create random inputs Q, K, V
                # Shape: (Batch, Seq, Head_Dim) -> 'Removing head dimension' means treating it as (B, S, D)
                Q = torch.randn(BATCH_SIZE, seq_len, d, device=device, requires_grad=True)
                K = torch.randn(BATCH_SIZE, seq_len, d, device=device, requires_grad=True)
                V = torch.randn(BATCH_SIZE, seq_len, d, device=device, requires_grad=True)

                # -------------------------------------------------------
                # (d) Forward Pass Benchmark
                # -------------------------------------------------------
                # Warmup
                for _ in range(WARMUP):
                    _ = annotated_scaled_dot_product_attention(Q, K, V)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                
                # Timing
                start_time = time.time()
                for _ in range(NUM_ITER):
                    _ = annotated_scaled_dot_product_attention(Q, K, V)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                
                fwd_time_ms = ((time.time() - start_time) / NUM_ITER) * 1000

                # -------------------------------------------------------
                # (e) Memory Measurement
                # -------------------------------------------------------
                # We need to measure memory "in use before the backward pass starts".
                # This equals the memory of the output + the activations saved for backward.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                # Run one forward pass to generate the graph and saved tensors
                out = annotated_scaled_dot_product_attention(Q, K, V)
                
                if torch.cuda.is_available():
                    mem_active = torch.cuda.memory_allocated()
                else:
                    mem_active = 0 # Cannot reliably measure on CPU in same way
                
                # -------------------------------------------------------
                # (e) Backward Pass Benchmark
                # -------------------------------------------------------
                grad_output = torch.randn_like(out)
                
                # Warmup Backward
                # (We must re-run forward every time to rebuild the graph for backward)
                for _ in range(WARMUP):
                    temp_out = annotated_scaled_dot_product_attention(Q, K, V)
                    temp_out.backward(grad_output)
                    Q.grad = None; K.grad = None; V.grad = None
                if torch.cuda.is_available(): torch.cuda.synchronize()

                # Timing Backward
                # We time ONLY the backward() call, but we must run forward inside the loop 
                # to allow backward to function.
                bwd_times = []
                for _ in range(NUM_ITER):
                    # Setup (untimed)
                    temp_out = annotated_scaled_dot_product_attention(Q, K, V)
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    
                    # Critical Section
                    t0 = time.time()
                    temp_out.backward(grad_output)
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    bwd_times.append(time.time() - t0)
                    
                    # Cleanup gradients
                    Q.grad = None; K.grad = None; V.grad = None

                bwd_time_ms = (sum(bwd_times) / len(bwd_times)) * 1000

                print(f"Fwd: {fwd_time_ms:.2f}ms | Bwd: {bwd_time_ms:.2f}ms | Mem: {format_memory(mem_active)}")
                
                results.append({
                    "Batch Size": BATCH_SIZE,
                    "d_head": d,
                    "seq_len": seq_len,
                    "Forward (ms)": f"{fwd_time_ms:.4f}",
                    "Backward (ms)": f"{bwd_time_ms:.4f}",
                    "Memory (Bytes)": mem_active,
                    "Memory (Formatted)": format_memory(mem_active)
                })
                
                # Cleanup
                del Q, K, V, out, grad_output, temp_out
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("OOM ERROR")
                    results.append({
                        "Batch Size": BATCH_SIZE,
                        "d_head": d,
                        "seq_len": seq_len,
                        "Forward (ms)": "OOM",
                        "Backward (ms)": "OOM",
                        "Memory (Bytes)": "OOM",
                        "Memory (Formatted)": "OOM"
                    })
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                else:
                    print(f"Runtime Error: {e}")
                    raise e

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print("\nBenchmark Complete.")
    print(f"Results saved to: {os.path.abspath(csv_path)}")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    run_benchmark()