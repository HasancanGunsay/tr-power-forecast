# PowerShell equivalent of the Makefile, because `make` is not available on a
# stock Windows install. Both files must stay in sync: the Makefile is what CI
# and Linux contributors use, this is what runs locally.
#
# Usage:  .\make.ps1 setup | lint | format | typecheck | test | check | clean

param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'setup', 'lint', 'format', 'typecheck', 'test', 'check', 'clean')]
    [string]$Target = 'help'
)

$ErrorActionPreference = 'Stop'

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

switch ($Target) {
    'help' {
        Write-Host @"
  setup       Create the locked environment and install git hooks
  lint        Check style and catch likely bugs
  format      Auto-format and auto-fix
  typecheck   Static type analysis
  test        Run the test suite with coverage
  check       Everything CI runs, locally
  clean       Remove caches and build artifacts
"@
    }
    'setup' {
        Invoke-Step 'uv sync' { uv sync --all-groups }
        Invoke-Step 'pre-commit install' { uv run pre-commit install }
    }
    'lint' { Invoke-Step 'ruff check' { uv run ruff check . } }
    'format' {
        Invoke-Step 'ruff format' { uv run ruff format . }
        Invoke-Step 'ruff check --fix' { uv run ruff check --fix . }
    }
    'typecheck' { Invoke-Step 'mypy' { uv run mypy } }
    'test' { Invoke-Step 'pytest' { uv run pytest } }
    'check' {
        Invoke-Step 'ruff check' { uv run ruff check . }
        Invoke-Step 'mypy' { uv run mypy }
        Invoke-Step 'pytest' { uv run pytest }
        Write-Host "All checks passed." -ForegroundColor Green
    }
    'clean' {
        foreach ($d in '.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov', 'dist', 'build') {
            if (Test-Path $d) { Remove-Item -Recurse -Force $d }
        }
        if (Test-Path '.coverage') { Remove-Item -Force '.coverage' }
        Write-Host "Cleaned." -ForegroundColor Green
    }
}
