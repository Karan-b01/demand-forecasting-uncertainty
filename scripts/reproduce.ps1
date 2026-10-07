$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
  throw "Project environment not found. Create it and install requirements using the README setup steps."
}

$requiredData = @(
  "data\raw\sales_train_evaluation.csv",
  "data\raw\calendar.csv",
  "data\raw\sell_prices.csv"
)
foreach ($relativePath in $requiredData) {
  if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $relativePath))) {
    throw "Required M5 file is missing: $relativePath. Download the dataset and follow README.md."
  }
}

Push-Location $repoRoot
try {
  $steps = @(
    "src.prepare_m5",
    "src.features",
    "src.train",
    "src.calibrate",
    "src.evaluate",
    "src.inventory",
    "src.failure_analysis"
  )
  foreach ($step in $steps) {
    Write-Host "`n=== python -m $step ===" -ForegroundColor Cyan
    & $pythonExe -m $step
    if ($LASTEXITCODE -ne 0) {
      throw "Pipeline step failed: python -m $step (exit code $LASTEXITCODE)"
    }
  }
  Write-Host "`nReproduction complete. Forecast outputs are in data/processed/ and diagnostics are in reports/." -ForegroundColor Green
}
finally {
  Pop-Location
}
