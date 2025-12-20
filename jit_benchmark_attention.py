import torch
import time
import pandas as pd
import os
import sys
import traceback
from cs336_systems.scaled_dot_product_attention import annotated_scaled_dot_product_attention

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BATCH_SIZE = 8
HEAD_DIMS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
NUM_ITER = 100
WARMUP = 10
OUTPUT_FILE = "attention_jit_results.csv"

def benchmark_func(func, Q, K, V, tag=""):
    """Runs timing benchmark for a specific function."""
    
    # 1. Warmup
    try:
        for _ in range(WARMUP):
            out = func(Q, K, V)
            grad = torch.randn_like(out)
            out.backward(grad)
            Q.grad = None; K.grad = None; V.grad = None
        torch.cuda.synchronize()
    except Exception as e:
        # If compilation fails (e.g. Triton missing), re-raise to catch in main loop
        raise RuntimeError(f"Warmup failed for {tag}: {e}")

    # 2. Measure Forward
    start_fwd = time.time()
    for _ in range(NUM_ITER):
        _ = func(Q, K, V)
    torch.cuda.synchronize()
    fwd_ms = ((time.time() - start_fwd) / NUM_ITER) * 1000

    # 3. Measure Backward
    grad_out = torch.randn_like(out)
    bwd_times = []
    
    for _ in range(NUM_ITER):
        # Re-run forward (untimed) to build graph
        out_temp = func(Q, K, V)
        torch.cuda.synchronize()
        
        # Time Backward
        t0 = time.time()
        out_temp.backward(grad_out)
        torch.cuda.synchronize()
        bwd_times.append(time.time() - t0)
        
        # Zero grads
        Q.grad = None; K.grad = None; V.grad = None

    bwd_ms = (sum(bwd_times) / len(bwd_times)) * 1000
    
    return fwd_ms, bwd_ms

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running JIT Attention Benchmark on {device}...")
    
    # Attempt to compile
    # On Windows, 'inductor' (default) often fails due to missing Triton.
    # We will try it, but catch errors gracefully.
    try:
        compiled_attention = torch.compile(annotated_scaled_dot_product_attention)
    except Exception as e:
        print(f"WARNING: torch.compile setup failed: {e}")
        compiled_attention = None

    results = []

    for d in HEAD_DIMS:
        for seq_len in SEQ_LENS:
            print(f"Config: B={BATCH_SIZE}, Seq={seq_len}, D={d}...", end=" ")
            
            # Helper to clean memory safely
            def clean_memory():
                if 'Q' in locals(): del Q
                if 'K' in locals(): del K
                if 'V' in locals(): del V
                torch.cuda.empty_cache()

            try:
                # Create Inputs
                Q = torch.randn(BATCH_SIZE, seq_len, d, device=device, requires_grad=True)
                K = torch.randn(BATCH_SIZE, seq_len, d, device=device, requires_grad=True)
                V = torch.randn(BATCH_SIZE, seq_len, d, device=device, requires_grad=True)
                
                # --- Benchmark Vanilla ---
                vanilla_fwd, vanilla_bwd = benchmark_func(
                    annotated_scaled_dot_product_attention, Q, K, V, tag="Vanilla"
                )
                
                # --- Benchmark Compiled ---
                jit_fwd, jit_bwd = 0.0, 0.0
                jit_status = "N/A"
                
                try:
                    jit_fwd, jit_bwd = benchmark_func(
                        compiled_attention, Q, K, V, tag="Compiled"
                    )
                    jit_status = "OK"
                    print(f"Done. Vanilla: {vanilla_fwd:.2f}ms | JIT: {jit_fwd:.2f}ms")
                except RuntimeError as e:
                    if "triton" in str(e).lower():
                        jit_status = "No Triton"
                        print("Failed (No Triton)", end=" ")
                    elif "out of memory" in str(e).lower():
                        raise e # Re-raise OOM to outer block
                    else:
                        jit_status = "Compile Err"
                        print(f"Failed ({e})", end=" ")

                results.append({
                    "d_head": d,
                    "seq_len": seq_len,
                    "Vanilla Fwd (ms)": f"{vanilla_fwd:.3f}",
                    "Vanilla Bwd (ms)": f"{vanilla_bwd:.3f}",
                    "Compiled Fwd (ms)": f"{jit_fwd:.3f}" if jit_status == "OK" else jit_status,
                    "Compiled Bwd (ms)": f"{jit_bwd:.3f}" if jit_status == "OK" else jit_status,
                    "Speedup Fwd": f"{vanilla_fwd/jit_fwd:.2f}x" if jit_fwd > 0 else "-",
                })

            except RuntimeError as e:
                # Catch OOM or other Critical Errors
                err_msg = str(e)
                if "out of memory" in err_msg:
                    print("OOM")
                    results.append({
                        "d_head": d,
                        "seq_len": seq_len,
                        "Vanilla Fwd (ms)": "OOM",
                        "Compiled Fwd (ms)": "OOM"
                    })
                    # Critical: Clear the error state
                    try:
                        clean_memory()
                    except:
                        pass 
                else:
                    print(f"Error: {e}")
            
            # Clean up after every iteration
            try:
                del Q, K, V
            except: 
                pass
            torch.cuda.empty_cache()

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    run_benchmark()