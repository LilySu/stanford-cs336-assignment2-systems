# Configuration
$OutputDir = ".\nsys_profiles_win"
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null }

$ModelSizes = @("small", "medium")
$ContextLengths = @(128, 256, 512, 1024)
$Modes = @("forward", "forward_backward", "full_training")

# --- AUTO-DETECT NSYS PATH ---
$NsysExe = "nsys"
if (-not (Get-Command "nsys" -ErrorAction SilentlyContinue)) {
    Write-Host "Searching for nsys.exe in standard locations..." -ForegroundColor Yellow
    
    # Look in Program Files for Nsight Systems folders, sorted by latest version
    $candidates = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Systems *" -ErrorAction SilentlyContinue | Sort-Object Name -Descending
    
    $found = $false
    foreach ($folder in $candidates) {
        $possiblePath = Join-Path $folder.FullName "target-windows-x64\nsys.exe"
        if (Test-Path $possiblePath) {
            $NsysExe = $possiblePath
            $found = $true
            Write-Host "Found nsys at: $NsysExe" -ForegroundColor Cyan
            break
        }
    }
    
    if (-not $found) {
        Write-Error "ERROR: Could not find 'nsys.exe'. Please install NVIDIA Nsight Systems for Windows."
        exit 1
    }
}

Write-Host "Starting Profiling..." -ForegroundColor Green

foreach ($model in $ModelSizes) {
    foreach ($ctx in $ContextLengths) {
        # Optional: Skip medium ctx=1024 if OOM occurs
        if ($model -eq "medium" -and $ctx -eq 1024) { 
            Write-Warning "Skipping medium ctx=1024 (optional)"
            # continue 
        }

        foreach ($mode in $Modes) {
            $testName = "${model}_ctx${ctx}_${mode}"
            $outputPath = Join-Path $OutputDir $testName
            
            Write-Host "------------------------------------------------"
            Write-Host "Profiling: $testName"
            
            # Run Nsight Systems
            $proc = Start-Process -FilePath $NsysExe -ArgumentList "profile", "-t", "cuda,nvtx", "--sample=none", "--cpuctxsw=none", "-o", "`"$outputPath`"", "--force-overwrite", "true", "python", "profile_benchmark.py", "--model_size", "$model", "--context_length", "$ctx", "--mode", "$mode", "--num_iterations", "10" -Wait -PassThru -NoNewWindow
            
            if ($proc.ExitCode -ne 0) {
                Write-Error "Profiling failed for $testName"
                continue
            }

            # Generate CSV Stats
            if (Test-Path "$outputPath.nsys-rep") {
                Write-Host "  Generating Kernel CSV..."
                $csvPath = "$outputPath`_kernels.csv"
                $sqlitePath = "$outputPath.sqlite"
                
                # Use cmd /c for reliable redirection
                cmd /c "`"$NsysExe`" stats --report cuda_gpu_kern_sum --format csv --force-export true `"$outputPath.nsys-rep`" > `"$csvPath`""
                
                # CLEANUP: Delete the unwanted .sqlite file immediately
                if (Test-Path $sqlitePath) {
                    Remove-Item $sqlitePath -Force -ErrorAction SilentlyContinue
                    Write-Host "  [CLEANUP] Deleted intermediate .sqlite file." -ForegroundColor Gray
                }
                
                if ((Get-Item $csvPath).Length -gt 300) {
                    Write-Host "  [SUCCESS] CSV generated." -ForegroundColor Cyan
                } else {
                    Write-Warning "  [WARNING] CSV is small. Check if GPU was used."
                }
            }
            Start-Sleep -Seconds 1
        }
    }
}
Write-Host "Done! Results in $OutputDir" -ForegroundColor Green