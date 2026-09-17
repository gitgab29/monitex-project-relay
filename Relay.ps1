<#
.SYNOPSIS
    Project Relay operator console. Run this, pick a number.

.DESCRIPTION
    Everything you need to operate the system, without remembering any paths or flags.
    Long-running things (the dashboard, the live camera) open in their OWN window so you can
    run them at the same time and still come back to this menu.

.EXAMPLE
    .\Relay.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# Everything resolves from this file's location, so it does not matter where you run it from.
$Root  = $PSScriptRoot
$Relay = Join-Path $Root '.venv\Scripts\relay.exe'
$Py    = Join-Path $Root '.venv\Scripts\python.exe'
$Db    = Join-Path $Root 'data\relay.db'
$Port  = 8080

function Write-Title {
    Clear-Host
    Write-Host ''
    Write-Host '  PROJECT RELAY' -ForegroundColor Cyan
    Write-Host '  camera feed -> structured events -> routed actions' -ForegroundColor DarkGray
    Write-Host ''
}

function Test-Setup {
    if (-not (Test-Path $Relay)) {
        Write-Host "  SETUP NOT FINISHED" -ForegroundColor Red
        Write-Host "  Could not find: $Relay" -ForegroundColor DarkGray
        Write-Host ""
        Write-Host "  Run these two lines once, then start me again:" -ForegroundColor Yellow
        Write-Host "    python -m venv .venv" -ForegroundColor White
        Write-Host "    .\.venv\Scripts\python.exe -m pip install -e `".[dev]`"" -ForegroundColor White
        Write-Host ""
        return $false
    }
    return $true
}

function Get-Status {
    # Is the dashboard already up?
    $api = 'not running'
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:$Port/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $api = "running on :$Port" }
    } catch { }

    $events = 'database is empty'
    if (Test-Path $Db) {
        try {
            $j = & $Relay summary --json 2>$null | ConvertFrom-Json
            $events = "$($j.totals.events) events, $($j.totals.needs_review) need review"
        } catch { $events = 'database present' }
    }
    return @{ Api = $api; Events = $events }
}

function Show-Menu {
    Write-Title
    $s = Get-Status
    Write-Host "  dashboard : $($s.Api)"      -ForegroundColor DarkGray
    Write-Host "  database  : $($s.Events)"   -ForegroundColor DarkGray
    Write-Host ''
    Write-Host '  WATCH' -ForegroundColor Yellow
    Write-Host '   1  Open the dashboard        (new window + your browser)'
    Write-Host '   2  Start the live camera     (new window - THIS IS THE SYSTEM)'
    Write-Host ''
    Write-Host '  LOOK AT RECORDS' -ForegroundColor Yellow
    Write-Host '   3  Summary of everything'
    Write-Host '   4  Review queue  (what needs a human)'
    Write-Host '   5  Resolve a review item'
    Write-Host '   6  Recorded sessions'
    Write-Host ''
    Write-Host '  BEFORE THE DEMO' -ForegroundColor Yellow
    Write-Host '   7  Check the Gemini key and Gmail password still work'
    Write-Host '   8  WIPE the database  (fresh start for recording)'
    Write-Host ''
    Write-Host '  PROVE IT WORKS' -ForegroundColor Yellow
    Write-Host '   9  Replay a recording        (same events, nothing duplicated)'
    Write-Host '  10  Show a failure recovering (--chaos timeout)'
    Write-Host '  11  Run the tests'
    Write-Host ''
    Write-Host '   Q  Quit' -ForegroundColor DarkGray
    Write-Host ''
}

function Start-InNewWindow ([string]$Title, [string]$Exe, [string[]]$Args) {
    # -NoExit so the window stays open and you can read what happened.
    $argLine = ($Args | ForEach-Object { if ($_ -match '\s') { "'$_'" } else { $_ } }) -join ' '
    $cmd = "`$host.UI.RawUI.WindowTitle='$Title'; & '$Exe' $argLine"
    Start-Process powershell -ArgumentList '-NoExit', '-NoProfile', '-Command', $cmd
}

function Pause-Here {
    Write-Host ''
    Write-Host '  press Enter to go back to the menu' -ForegroundColor DarkGray
    [void](Read-Host)
}

function Pick-Session {
    $dir = Join-Path $Root 'data\sessions'
    if (-not (Test-Path $dir)) { return $null }
    $files = @(Get-ChildItem $dir -Filter *.mp4 | Sort-Object LastWriteTime -Descending)
    if ($files.Count -eq 0) { return $null }
    Write-Host ''
    for ($i = 0; $i -lt $files.Count; $i++) {
        $mb = [math]::Round($files[$i].Length / 1MB, 1)
        Write-Host ("   [{0}] {1}  ({2} MB)" -f ($i + 1), $files[$i].Name, $mb)
    }
    Write-Host ''
    $n = Read-Host '  which one (number)'
    $idx = 0
    if ([int]::TryParse($n, [ref]$idx) -and $idx -ge 1 -and $idx -le $files.Count) {
        return $files[$idx - 1].FullName
    }
    return $null
}

# ------------------------------------------------------------------ main loop

if (-not (Test-Setup)) { Pause-Here; exit 1 }

while ($true) {
    Show-Menu
    $choice = Read-Host '  pick one'

    switch ($choice.Trim().ToUpper()) {

        '1' {
            Start-InNewWindow 'RELAY - dashboard' $Relay @('api', '--port', "$Port")
            Write-Host ''
            Write-Host '  starting the dashboard...' -ForegroundColor DarkGray
            $up = $false
            foreach ($i in 1..15) {
                Start-Sleep -Milliseconds 400
                try {
                    $r = Invoke-WebRequest -Uri "http://localhost:$Port/health" -TimeoutSec 2 -UseBasicParsing
                    if ($r.StatusCode -eq 200) { $up = $true; break }
                } catch { }
            }
            if ($up) {
                Start-Process "http://localhost:$Port/summary?format=html"
                Write-Host '  dashboard open in your browser.' -ForegroundColor Green
                Write-Host "    dashboard    http://localhost:$Port/summary?format=html"
                Write-Host "    API explorer http://localhost:$Port/docs"
                Write-Host '  leave that new window open while you use the system.' -ForegroundColor DarkGray
            }
            else {
                Write-Host '  it did not come up. Check the new window for the error.' -ForegroundColor Red
            }
            Pause-Here
        }

        '2' {
            Write-Host ''
            Write-Host '  Opening the live camera in a new window.' -ForegroundColor Green
            Write-Host '  A video window will appear with the zones drawn on it.' -ForegroundColor DarkGray
            Write-Host '  Sit at the desk    -> post_manned' -ForegroundColor DarkGray
            Write-Host '  Walk away 8 sec    -> post_unattended, HIGH, you get an email' -ForegroundColor DarkGray
            Write-Host '  Press q in the video window to stop.' -ForegroundColor DarkGray
            Start-InNewWindow 'RELAY - LIVE' $Relay @('run', '--source', 'webcam', '--show')
            Pause-Here
        }

        '3' { Write-Title; & $Relay summary; Pause-Here }

        '4' {
            Write-Title
            & $Relay review list
            Write-Host ''
            Write-Host '  use option 5 to close one of these off.' -ForegroundColor DarkGray
            Pause-Here
        }

        '5' {
            Write-Title
            & $Relay review list
            Write-Host ''
            $id = Read-Host '  event_id to resolve (blank to cancel)'
            if ($id.Trim() -ne '') {
                Write-Host ''
                Write-Host '   [1] confirmed    it was real'
                Write-Host '   [2] false_alarm  nothing happened'
                Write-Host '   [3] dismissed    not worth recording'
                Write-Host ''
                $k = Read-Host '  which'
                $as = $null
                if ($k -eq '1') { $as = 'confirmed' }
                if ($k -eq '2') { $as = 'false_alarm' }
                if ($k -eq '3') { $as = 'dismissed' }
                if ($as) {
                    $notes = Read-Host '  notes (optional)'
                    if ($notes.Trim() -eq '') { & $Relay review resolve $id.Trim() --as $as }
                    else { & $Relay review resolve $id.Trim() --as $as --notes $notes }
                }
                else { Write-Host '  cancelled.' -ForegroundColor DarkGray }
            }
            Pause-Here
        }

        '6' { Write-Title; & $Relay sessions list; Pause-Here }

        '7' {
            Write-Title
            & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'scripts\Verify-Credentials.ps1')
            Pause-Here
        }

        '8' {
            Write-Title
            Write-Host '  This deletes every event, review item and dead letter.' -ForegroundColor Yellow
            Write-Host '  Your recorded videos in data\sessions are NOT touched.' -ForegroundColor DarkGray
            Write-Host ''
            $c = Read-Host '  type WIPE to confirm'
            if ($c -ceq 'WIPE') {
                if (Test-Path $Db) {
                    Remove-Item $Db -Force
                    Get-ChildItem (Split-Path $Db) -Filter 'relay.db-*' -EA SilentlyContinue | Remove-Item -Force
                    Write-Host '  database deleted. It rebuilds empty on the next run.' -ForegroundColor Green
                }
                else { Write-Host '  there was no database to delete.' -ForegroundColor DarkGray }
            }
            else { Write-Host '  cancelled, nothing deleted.' -ForegroundColor DarkGray }
            Pause-Here
        }

        '9' {
            Write-Title
            Write-Host '  Replaying a recording proves re-running it does NOT duplicate events.' -ForegroundColor DarkGray
            $f = Pick-Session
            if ($f) { Write-Host ''; & $Relay replay $f --no-n8n }
            else { Write-Host '  no recordings yet - use option 2 first.' -ForegroundColor Yellow }
            Pause-Here
        }

        '10' {
            Write-Title
            Write-Host '  Forcing the model to time out, to show it retry, back off and degrade.' -ForegroundColor DarkGray
            $f = Pick-Session
            if ($f) {
                $sid = 'chaos-' + (Get-Date -Format 'HHmmss')
                Write-Host ''
                & $Relay replay $f --chaos timeout --session-id $sid --no-n8n
            }
            else { Write-Host '  no recordings yet - use option 2 first.' -ForegroundColor Yellow }
            Pause-Here
        }

        '11' { Write-Title; & $Py -m pytest -q -p no:warnings; Pause-Here }

        'Q' { Write-Host ''; Write-Host '  bye.' -ForegroundColor DarkGray; Write-Host ''; exit 0 }

        default { }
    }
}
