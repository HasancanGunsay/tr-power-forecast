# Running this on a schedule

`daily.ps1` performs one scheduled run: ingest → retrain if due → forecast
tomorrow → verify. It is the orchestrator; each job it calls is independently
runnable and reports its own exit code.

**Timing.** Bids for delivery day D close at 12:30 on D−1 and the forecast origin
is 11:00, so the run belongs between those. **11:30** leaves an hour of slack for
a slow ingest without approaching the deadline.

**The contract.** The script exits 0 only if tomorrow's forecast was stored.
Ingestion failing on a platform outage is logged and does not end the run — the
panel may still reach far enough to forecast from, and if it does not, the daily
job refuses the day loudly and the exit code says so.

**Logs** land in `logs/YYYY-MM-DD.log`, one file a day, appended. Gitignored.

## Try it once before scheduling it

```powershell
.\scripts\daily.ps1
```

Read the log it writes. A scheduled task that has never been run by hand is a
scheduled task nobody has debugged.

## Register the task

Run this once, in a PowerShell window. It creates a task under your own account
that runs every day at 11:30.

```powershell
$repo = "C:\Users\hasan\Desktop\Hasancan_Self_Improve\tr-power-forecast"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$repo\scripts\daily.ps1`"" -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Daily -At 11:30
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName "tr-power-forecast daily" -Action $action -Trigger $trigger -Settings $settings -Description "Ingest, retrain if due, forecast tomorrow, verify."
```

Two settings worth understanding rather than copying:

- **`-StartWhenAvailable`** runs the task late if the machine was asleep at
  11:30. Without it a closed laptop simply means no forecast that day, and
  nothing says so. With it, a late run still produces the day — and the log
  timestamp shows it was late.
- **`-ExecutionTimeLimit 1 hour`** kills a run that hangs. A backfill waiting on
  an unresponsive API would otherwise still be holding the task open when
  tomorrow's run is due, and Task Scheduler would skip that one.

## Checking on it

```powershell
Get-ScheduledTaskInfo -TaskName "tr-power-forecast daily"
```

`LastTaskResult` is the script's exit code: **0 means tomorrow's forecast exists.**
Anything else, read `logs\` for the day.

Run it now without waiting for 11:30:

```powershell
Start-ScheduledTask -TaskName "tr-power-forecast daily"
```

Remove it:

```powershell
Unregister-ScheduledTask -TaskName "tr-power-forecast daily" -Confirm:$false
```

## What this does not do

**Nothing tells you when it fails.** `LastTaskResult` records it and the log
explains it, but both wait to be read. On a personal machine that is the honest
level to stop at — an alerting path that is never wired to anything real is
decoration. If this ever matters, the smallest useful step is the script posting
its exit code somewhere you already look.

**The machine has to be awake.** `-StartWhenAvailable` covers a late start, not a
laptop that stayed shut all day. A missed day is visible as a gap in the forecast
store and in the monitoring report, which is the right place for it to show up.
