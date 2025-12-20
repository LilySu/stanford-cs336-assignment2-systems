import torch
import timeit
import statistics
import pandas as pd

def get_model_specs():
    """
    Returns the dictionary of model specifications (Table 1) from the assignment.
    """
    return {
        "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
        "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
        "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
        "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
        "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    }

def generate_batch(vocab_size, batch_size, context_length, device):
    """
    Generates a random batch of integer inputs for the model.
    """
    return torch.randint(
        0, vocab_size, (batch_size, context_length), 
        device=device, dtype=torch.int64
    )

def benchmark_pass(model, input_ids, mode, warmup_steps=5, num_steps=10):
    """
    Performs the timing loop for a specific mode (forward or backward).
    
    Args:
        model: The initialized Transformer model.
        input_ids: The input tensor batch.
        mode: "forward" or "backward".
        warmup_steps: Number of steps to run before timing.
        num_steps: Number of steps to measure.
        
    Returns:
        (avg_ms, std_ms): Tuple of average time and standard deviation in milliseconds.
    """
    device = input_ids.device
    
    # 1. Define the Step Function based on mode
    if mode == "backward":
        model.train()
        for param in model.parameters():
            param.requires_grad = True
            
        def step_fn():
            model.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = logits.sum()
            loss.backward()
    else:
        model.eval()
        def step_fn():
            with torch.no_grad():
                _ = model(input_ids)

    # 2. Warmup Phase
    for _ in range(warmup_steps):
        step_fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # 3. Measurement Phase
    timer = timeit.default_timer
    timings = []

    for _ in range(num_steps):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        t0 = timer()
        step_fn()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        t1 = timer()
        timings.append(t1 - t0)

    # 4. Calculate Statistics
    avg_ms = statistics.mean(timings) * 1000
    std_ms = statistics.stdev(timings) * 1000 if len(timings) > 1 else 0.0
    
    return avg_ms, std_ms

def print_results_table(results_list):
    """
    Formats the collected results into Markdown and LaTeX tables using Pandas.
    """
    df = pd.DataFrame(results_list)
    
    print("\n" + "="*20 + " MARKDOWN TABLE " + "="*20)
    print(df.to_markdown(index=False))
    
    print("\n" + "="*20 + " LATEX TABLE " + "="*20)
    print(df.to_latex(index=False, float_format="%.2f", caption="Benchmark Results", label="tab:benchmarks"))