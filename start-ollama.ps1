#Requires -Version 5.1
<#
  start-ollama.ps1
  Start Ollama on CPU so the GPU is reserved for the bge-reranker-v2-m3
  cross-encoder in the RAG backend.

  Why: the RTX 3050 Ti Laptop has 4GB VRAM. bge-m3 (Ollama embed) and
  bge-reranker-v2-m3 (PyTorch reranker) do not fit together. If Ollama
  grabs the GPU, the reranker spills to shared RAM and rerank goes from
  ~15s to 50-150s. Pinning Ollama to CPU keeps the reranker on the GPU.

  Usage:
    .\start-ollama.ps1              # (re)start Ollama on CPU
    .\start-ollama.ps1 -Force       # stop a running Ollama without prompting
    .\start-ollama.ps1 -DryRun      # show what would happen, change nothing
#>
[CmdletBinding()]
param(
  [switch]$Force,
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

function Resolve-OllamaExe {
  $c = Get-Command ollama -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  $known = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
  if (Test-Path $known) { return $known }
  throw "ollama.exe not found on PATH or at $known"
}

function Stop-RunningOllama {
  $procs = Get-Process ollama -ErrorAction SilentlyContinue
  if (-not $procs) { return $false }
  if (-not $Force) {
    Write-Host "Ollama is running. Use -Force to restart it on CPU." -ForegroundColor Yellow
    $r = Read-Host "Restart now? [y/N]"
    if ($r -ne 'y' -and $r -ne 'Y') {
      Write-Host "Leaving existing Ollama running. Exiting." -ForegroundColor Gray
      exit 0
    }
  }
  foreach ($p in $procs) {
    try { taskkill /T /PID $p.Id /F 2>&1 | Out-Null } catch { }
    Start-Sleep -Milliseconds 200
  }
  Start-Sleep -Seconds 2
  if (Get-Process ollama -ErrorAction SilentlyContinue) {
    throw "Failed to stop running Ollama."
  }
  return $true
}

function Wait-OllamaUp($timeoutSec = 30) {
  $deadline = (Get-Date).AddSeconds($timeoutSec)
  while ((Get-Date) -lt $deadline) {
    try {
      $r = Invoke-WebRequest -UseBasicParsing http://localhost:11434/api/tags -TimeoutSec 3
      if ($r.StatusCode -eq 200) { return $true }
    } catch { }
    Start-Sleep -Seconds 1
  }
  return $false
}

function Get-GpuMemMib {
  $line = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>$null)
  if (-not $line) { return 0 }
  if ($line -match '(\d+)\s*MiB') { return [int]$Matches[1] }
  return 0
}

function Test-OllamaOnCpu {
  $before = Get-GpuMemMib
  try {
    Invoke-WebRequest -UseBasicParsing -Method Post -Uri http://localhost:11434/api/embeddings `
      -Body '{"model":"bge-m3","prompt":"health-check"}' -ContentType 'application/json' -TimeoutSec 120 | Out-Null
  } catch {
    Write-Warning "embed call failed (model may need pulling): $($_.Exception.Message)"
    return $null
  }
  Start-Sleep -Seconds 1
  $after = Get-GpuMemMib
  return $after - $before
}

# --- main ---
$exe = Resolve-OllamaExe
Write-Host "Ollama exe: $exe"

if ($DryRun) {
  Write-Host "DryRun: would stop any running Ollama, then start with:" -ForegroundColor Cyan
  Write-Host "  CUDA_VISIBLE_DEVICES = ''" -ForegroundColor Cyan
  Write-Host "  OLLAMA_NUM_GPU       = 0" -ForegroundColor Cyan
  Write-Host "  OLLAMA_HOST          = 127.0.0.1:11434" -ForegroundColor Cyan
  exit 0
}

$stopped = Stop-RunningOllama
if ($stopped) { Write-Host "Stopped existing Ollama." -ForegroundColor Gray }

$portInUse = Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue
if ($portInUse) {
  throw "Port 11434 still in use after stop. Aborting."
}

# Pin Ollama to CPU -- hide the GPU entirely.
$env:CUDA_VISIBLE_DEVICES = ''
$env:OLLAMA_NUM_GPU = '0'
$env:OLLAMA_HOST = '127.0.0.1:11434'

Write-Host "Starting Ollama on CPU (CUDA_VISIBLE_DEVICES='', OLLAMA_NUM_GPU=0)..." -ForegroundColor Cyan
$p = Start-Process -FilePath $exe -ArgumentList 'serve' -WindowStyle Hidden -PassThru
Write-Host "Started pid=$($p.Id)"

if (-not (Wait-OllamaUp 30)) {
  throw "Ollama did not come up on :11434 within 30s. Check logs."
}
Write-Host "Ollama listening on :11434" -ForegroundColor Green

$delta = Test-OllamaOnCpu
if ($null -ne $delta) {
  if ($delta -le 400) {
    Write-Host "GPU check: VRAM delta = $delta MiB (Ollama on CPU - reranker keeps the GPU)" -ForegroundColor Green
  } else {
    Write-Host "GPU check: VRAM delta = $delta MiB (WARNING: Ollama may still be using the GPU)" -ForegroundColor Yellow
    Write-Host "  If this persists, ensure no other Ollama process grabbed the GPU first." -ForegroundColor Yellow
  }
}

$usedMib = Get-GpuMemMib
$totalLine = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>$null)
$totalMib = if ($totalLine -match '(\d+)\s*MiB') { $Matches[1] } else { '?' }
Write-Host "GPU VRAM now: $usedMib / $totalMib MiB"
Write-Host "Done. Backend reranker has the GPU to itself." -ForegroundColor Green
