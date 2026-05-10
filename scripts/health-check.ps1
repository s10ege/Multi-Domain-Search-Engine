$python = "c:/Users/soner/.vscode/projects/123/.venv/Scripts/python.exe"
if (-not (Test-Path $python)) {
  throw "Python executable not found at $python"
}

$results = @()

function Run-Step {
  param(
    [string]$Label,
    [scriptblock]$Command
  )

  Write-Host $Label -ForegroundColor Cyan
  & $Command
  $exit = $LASTEXITCODE

  if ($exit -eq 0) {
    Write-Host "-> PASS" -ForegroundColor Green
    $script:results += [pscustomobject]@{ Step = $Label; Status = "PASS"; ExitCode = 0 }
  }
  else {
    Write-Host "-> FAIL (exit code $exit)" -ForegroundColor Red
    $script:results += [pscustomobject]@{ Step = $Label; Status = "FAIL"; ExitCode = $exit }
  }

  Write-Host ""
}

Run-Step "[1/4] Dependency check (pip check)" { & $python -m pip check }
Run-Step "[2/4] Test suite (pytest backend/test/ -q)" { & $python -m pytest backend/test/ -q }
Run-Step "[3/4] Lint check (ruff)" { & $python -m ruff check . }
Run-Step "[4/4] Type check (pyright)" { & $python -m pyright }

Write-Host "Summary:" -ForegroundColor Yellow
$results | Format-Table -AutoSize

if ($results.Status -contains "FAIL") {
  exit 1
}

Write-Host "Health check completed." -ForegroundColor Green
