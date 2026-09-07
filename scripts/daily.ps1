<#
.SYNOPSIS
    One scheduled run: ingest, retrain if due, forecast tomorrow, verify.

.DESCRIPTION
    Run before the 12:30 bid deadline. The forecast origin is 11:00 local, so
    11:30 leaves an hour of slack for a slow ingest without touching the deadline.

    This script is the orchestrator. Each job it calls is independently runnable
    and reports its own exit code; keeping them separate means one broken thing
    stops one thing.

    The contract: **this script exits 0 only if tomorrow's forecast was stored.**
    Everything else is logged and does not, on its own, decide the outcome —
    ingestion can fail on a platform outage while the panel still reaches far
    enough to forecast from, and a retrain that is not due is not a problem.

.NOTES
    Register it with Task Scheduler — see scripts/README.md. Logs land in
    logs/YYYY-MM-DD.log, one file a day, appended.
#>

[CmdletBinding()]
param(
    # Where the repository lives. Resolved from the script's own location when
    # omitted, so the scheduled task does not need a working directory — Task
    # Scheduler does not set one the way a shell would.
    [string]$RepoRoot
)

# Resolved here rather than as a param default: $PSScriptRoot is not reliably
# populated while the param block is being evaluated, and the failure is an
# empty string rather than an error, which then resolves to the wrong directory.
if (-not $RepoRoot) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $RepoRoot = Split-Path -Parent $scriptDir
}

Set-Location -Path $RepoRoot

$logDir = Join-Path $RepoRoot 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir ("{0:yyyy-MM-dd}.log" -f (Get-Date))

# Note on both helpers: nothing may be left on the pipeline.
#
# A PowerShell function returns *everything* written to the pipeline inside it,
# not just what `return` names. `Tee-Object` writes to the pipeline as well as to
# the file, so an earlier version of `Invoke-Step` returned several kilobytes of
# job output with the exit code stuck on the end — and `exit $forecast` then
# exited on a string. The run looked successful while the forecast had failed.
#
# `Out-Host` prints without emitting, which is what keeps the return value an
# integer.

function Write-Log {
    param([string]$Message)
    $line = "{0:yyyy-MM-dd HH:mm:ss}  {1}" -f (Get-Date), $Message
    # UTF8 explicitly: PowerShell 5.1 writes UTF-16 by default, which makes the
    # log unreadable to grep and to anything expecting text.
    Add-Content -Path $logFile -Value $line -Encoding UTF8
    Write-Host $line
}

function Invoke-Step {
    <#
        Runs one job and records its exit code. Never throws: a scheduled run
        that dies on step two tells you nothing about steps three and four, and
        the whole point of the log is to be readable the morning after.
    #>
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    Write-Log "--> $Name"
    & uv run python @Arguments *>&1 | Out-String -Stream | Add-Content -Path $logFile -Encoding UTF8 -PassThru | Out-Host
    $code = $LASTEXITCODE
    Write-Log "<-- $Name exited $code"
    return $code
}

Write-Log "=== scheduled run starting in $RepoRoot ==="

# 1. Ingest. The trailing two months are re-fetched even though their files
#    exist, because a month written today is not finished (see data/backfill.py).
$ingest = Invoke-Step -Name 'ingest: EPIAS' -Arguments @('-m', 'powerforecast.data.backfill')
$weather = Invoke-Step -Name 'ingest: weather archive' -Arguments @('-m', 'powerforecast.data.backfill_weather')
$supply = Invoke-Step -Name 'ingest: supply weather archive' -Arguments @('-m', 'powerforecast.data.backfill_supply_weather')

if ($ingest -ne 0) {
    Write-Log "NOTE ingest failed ($ingest). Continuing: the panel may still reach far enough."
}
if ($weather -ne 0) {
    Write-Log "NOTE weather archive failed ($weather). Tomorrow's temperature comes from the live run, not this."
}
if ($supply -ne 0) {
    Write-Log "NOTE supply weather archive failed ($supply). Tomorrow's irradiance and wind come from the live run, not this."
}

# Both targets get the same treatment. Kept as a list rather than two copied
# blocks so that adding a third target is one string, and so a step added for
# one target cannot be forgotten for the other.
$targets = @('load', 'price')

# 2. Retrain if due, per target. Exit 4 means the panel has not advanced, which
#    is expected on a day the ingest found nothing new — not a failure.
foreach ($target in $targets) {
    $retrain = Invoke-Step -Name "retrain $target (if due)" -Arguments @('-m', 'powerforecast.jobs.retrain', '--target', $target)
    switch ($retrain) {
        0 { }
        4 { Write-Log "NOTE retrain $target skipped: no new data since the model was trained." }
        2 { Write-Log "WARN retrain $target refused to promote a candidate that regressed. The incumbent stays." }
        default { Write-Log "WARN retrain $target exited $retrain." }
    }
}

# 3. The thing that must happen — for every target.
#
#    The exit code stays narrow: zero only if *every* target's forecast reached
#    the store. A day where price was bid and load was not is not a successful
#    day, and collapsing that into "mostly worked" is how a missing forecast
#    goes unnoticed. The first non-zero is what the script exits with, and the
#    log names which target produced it.
$forecast = 0
foreach ($target in $targets) {
    $code = Invoke-Step -Name "forecast tomorrow: $target" -Arguments @('-m', 'powerforecast.jobs.daily_forecast', '--target', $target)
    if ($code -ne 0) {
        Write-Log "WARN forecast for $target exited $code."
        if ($forecast -eq 0) { $forecast = $code }
    }
}

# 4. Verify what past forecasts turned out to be worth. Informational: a
#    DEGRADED verdict is news, not a broken run, so it never decides the exit.
#
#    Note what "skill" means differs by target — against the operator plan for
#    load, against a seasonal naive for price, because no price plan is
#    published (ADR 0012). The verdict names its control.
foreach ($target in $targets) {
    $null = Invoke-Step -Name "verify past forecasts: $target" -Arguments @('-m', 'powerforecast.monitoring.verify', '--target', $target)
}

Write-Log "=== finished; forecast step exited $forecast ==="
exit $forecast
