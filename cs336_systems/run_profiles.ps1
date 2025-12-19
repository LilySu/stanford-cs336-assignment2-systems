# Configuration
$OutputDir = ".\nsys_profiles_win"
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null }

$ModelSizes = @("small", "medium", "large")
$Precisions = @("fp32", "bf16")

# --- Generate Unique Timestamp ---
$TimeStamp = Get-Date -Format "yyyyMMdd_HHmmss"

# --- AUTO-DETECT NSYS PATH ---
$NsysExe = "nsys"
if (-not (Get-Command "nsys" -ErrorAction SilentlyContinue)) {
    Write-Host "Searching for nsys.exe..." -ForegroundColor Yellow
    $candidates = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Systems *" -ErrorAction SilentlyContinue | Sort-Object Name -Descending
    foreach ($folder in $candidates) {
        $possiblePath = Join-Path $folder.FullName "target-windows-x64\nsys.exe"
        if (Test-Path $possiblePath) {
            $NsysExe = $possiblePath
            break
        }
    }
}

Write-Host "Starting Profiling (Run ID: $TimeStamp)..." -ForegroundColor Green

foreach ($model in $ModelSizes) {
    foreach ($prec in $Precisions) {
        
        # 1. Define Unique Output Name
        $testName = "${model}_forward_backward_${prec}_${TimeStamp}"
        $outputPath = Join-Path $OutputDir $testName

        # 2. Build the Argument List as a flat array
        # First, the standard Nsight Systems arguments
        $FullArgs = @(
            "profile",
            "-t", "cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "-o", "$outputPath",
            "--force-overwrite", "true",
            "python", "profile_benchmark.py",
            "--model_size", "$model",
            "--mode", "forward_backward",
            "--num_iterations", "10"
        )
        
        # 3. Add Mixed Precision flag if needed
        if ($prec -eq "bf16") {
            $FullArgs += "--mixed_precision"
        }
        
        Write-Host "------------------------------------------------"
        Write-Host "Profiling: $testName"

        # 4. Run Nsight Systems with the flat array
        $proc = Start-Process -FilePath $NsysExe -ArgumentList $FullArgs -Wait -PassThru -NoNewWindow
        
        if ($proc.ExitCode -ne 0) {
            Write-Error "Profiling failed for $testName"
            continue
        }

        # Generate CSV Stats
        if (Test-Path "$outputPath.nsys-rep") {
            Write-Host "  Generating Kernel CSV..."
            $csvPath = "$outputPath`_kernels.csv"
            $sqlitePath = "$outputPath.sqlite"
            
            # Export to CSV
            cmd /c "`"$NsysExe`" stats --report cuda_gpu_kern_sum --format csv --force-export true `"$outputPath.nsys-rep`" > `"$csvPath`""
            
            # Cleanup intermediate sqlite file
            if (Test-Path $sqlitePath) { Remove-Item $sqlitePath -Force -ErrorAction SilentlyContinue }
            
            Write-Host "  [SUCCESS] Saved to $csvPath" -ForegroundColor Cyan
        }
        Start-Sleep -Seconds 1
    }
}
Write-Host "Done! Results in $OutputDir" -ForegroundColor Green