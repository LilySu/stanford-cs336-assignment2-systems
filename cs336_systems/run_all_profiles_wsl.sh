#!/bin/bash

# Use Linux nsys (installed in WSL)
NSYS="nsys"

# Check if nsys exists
if ! command -v $NSYS &> /dev/null; then
    echo "ERROR: nsys not found. Please install it:"
    echo "sudo apt install -y nsight-systems-2024.6.2"
    exit 1
fi

# Check if nvidia-smi works (GPU visible)
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Ensure NVIDIA drivers are installed."
else
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
fi

echo "Found nsys: $(which $NSYS)"
echo "Version: $($NSYS --version | head -1)"
echo ""

# Get absolute path for output directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR="$SCRIPT_DIR/nsys_profiles"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/profiling_log.txt"
echo "Starting profiling experiments at $(date)" > "$LOG_FILE"

# Run nsys status to debug environment issues in the log
$NSYS status -e >> "$LOG_FILE" 2>&1

# Model configurations
MODEL_SIZES=("small" "medium")
CONTEXT_LENGTHS=(128 256 512 1024)
MODES=("forward" "forward_backward" "full_training")

# Function to run profiling
run_profile() {
    local model=$1
    local ctx=$2
    local mode=$3
    
    output_name="${model}_ctx${ctx}_${mode}"
    output_path="$OUTPUT_DIR/$output_name"
    
    echo "========================================"
    echo "Profiling: Model=$model, Context=$ctx, Mode=$mode"
    echo "Time: $(date)"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # FIXED COMMAND:
    # 1. Added --sample=none --cpuctxsw=none: Disables CPU sampling/context switches.
    #    This is critical for WSL to avoid overhead and "dropped event" errors that cause empty traces.
    # 2. Kept --trace=cuda,nvtx: Captures the GPU kernels and NVTX ranges.
    # 3. Added --cuda-memory-usage=true: Helps confirm CUDA hooks are active.
    $NSYS profile \
        -o "$output_path" \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        python3 profile_benchmark.py \
            --model_size "$model" \
            --context_length "$ctx" \
            --mode "$mode" \
            --num_iterations 10 2>&1 | tee -a "$LOG_FILE"
    
    exit_code=$?

    sleep 2
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Success: $output_name" | tee -a "$LOG_FILE"
        
        # Check if .nsys-rep was created
        if [ -f "${output_path}.nsys-rep" ]; then
            rep_size=$(stat -c%s "${output_path}.nsys-rep" 2>/dev/null || stat -f%z "${output_path}.nsys-rep")
            echo "  .nsys-rep file created: ${rep_size} bytes" | tee -a "$LOG_FILE"
            
            # Export to CSV
            echo "  Exporting kernel stats to CSV..." | tee -a "$LOG_FILE"
            $NSYS stats \
                --report cuda_gpu_kern_sum \
                --format csv \
                --force-export=true \
                "${output_path}.nsys-rep" \
                > "${output_path}_kernels.csv" 2>&1
            
            $NSYS stats \
                --report nvtx_sum \
                --format csv \
                --force-export=true \
                "${output_path}.nsys-rep" \
                > "${output_path}_nvtx.txt" 2>&1
            
            # Check if CSV has data
            # NOTE: Empty files are usually ~80-150 bytes (headers only). 
            # A valid file with kernels will be larger.
            csv_size=$(wc -c < "${output_path}_kernels.csv")
            if [ $csv_size -gt 300 ]; then
                echo "  ✓ Kernel data captured successfully ($csv_size bytes)" | tee -a "$LOG_FILE"
            else
                echo "  ⚠ Warning: CSV file is small ($csv_size bytes). Check NVIDIA Control Panel permissions." | tee -a "$LOG_FILE"
                echo "  CSV contents preview:" | tee -a "$LOG_FILE"
                head -5 "${output_path}_kernels.csv" | tee -a "$LOG_FILE"
            fi
        else
            echo "  ✗ Error: .nsys-rep file was not created" | tee -a "$LOG_FILE"
            return 1
        fi
    else
        echo "✗ Failed: $output_name (exit code: $exit_code)" | tee -a "$LOG_FILE"
        
        # Check for OOM
        if grep -q "out of memory" "$LOG_FILE"; then
            echo "  Reason: Out of Memory" | tee -a "$LOG_FILE"
            return 2  # Special exit code for OOM
        fi
        return 1
    fi
    
    echo "" | tee -a "$LOG_FILE"
    return 0
}

# Main profiling loop
total_experiments=0
successful_experiments=0
failed_experiments=0
oom_experiments=0

echo "Starting profiling..."
echo "Output directory: $OUTPUT_DIR"
echo ""

for model in "${MODEL_SIZES[@]}"; do
    echo ""
    echo "========================================"
    echo "PROFILING MODEL: $model"
    echo "========================================"
    
    model_oom=false
    
    for ctx in "${CONTEXT_LENGTHS[@]}"; do
        # Skip if we already OOM'd on a smaller context length
        if [ "$model_oom" = true ]; then
            echo "Skipping ctx=$ctx for model=$model (previous OOM)" | tee -a "$LOG_FILE"
            continue
        fi
        
        for mode in "${MODES[@]}"; do
            ((total_experiments++))
            
            run_profile "$model" "$ctx" "$mode"
            result=$?
            
            if [ $result -eq 0 ]; then
                ((successful_experiments++))
            elif [ $result -eq 2 ]; then
                # OOM error
                ((failed_experiments++))
                ((oom_experiments++))
                model_oom=true
                echo "Marking model=$model as OOM, skipping larger contexts" | tee -a "$LOG_FILE"
                break  # Break out of mode loop
            else
                ((failed_experiments++))
            fi
            
            # Small delay between runs
            sleep 2
        done
        
        if [ "$model_oom" = true ]; then
            break  # Break out of context length loop
        fi
    done
done

# Summary
echo ""
echo "========================================"
echo "PROFILING COMPLETE"
echo "========================================"
echo "Total experiments: $total_experiments"
echo "Successful: $successful_experiments" 
echo "Failed: $failed_experiments (OOM: $oom_experiments)"
echo "Output directory: $OUTPUT_DIR"
echo "Completed at: $(date)"
echo ""