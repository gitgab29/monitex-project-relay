<#
.SYNOPSIS
    Prove the Gemini key and the Gmail app password actually work.

.DESCRIPTION
    A native check that needs no venv, no pip install and no Python: it calls the Gemini REST
    endpoint and Gmail's SMTP server directly.

    This exists because every credential path in Project Relay degrades quietly on purpose. A
    wrong Gemini key and a working one produce identical output -- the pipeline just falls back
    to template summaries -- so the first honest signal would otherwise be the demo itself.

    Secrets are never printed, only their length and first few characters.

.PARAMETER Gemini
    Check Gemini only.

.PARAMETER Smtp
    Check Gmail only. NOTE: this sends a real email to ALERT_TO.

.PARAMETER EnvFile
    Path to the env file. Defaults to .env next to the repo root.

.EXAMPLE
    .\scripts\Verify-Credentials.ps1
    .\scripts\Verify-Credentials.ps1 -Gemini
    .\scripts\Verify-Credentials.ps1 -Smtp
#>
[CmdletBinding()]
param(
    [switch]$Gemini,
    [switch]$Smtp,
    [string]$EnvFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# PowerShell 5.1 still negotiates TLS 1.0 by default against some endpoints; Google refuses it.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Write-Ok   ([string]$m) { Write-Host "  [OK]   $m"   -ForegroundColor Green }
function Write-Bad  ([string]$m) { Write-Host "  [FAIL] $m"   -ForegroundColor Red }
function Write-Fix  ([string]$m) { Write-Host "         -> $m" -ForegroundColor Yellow }
function Write-Head ([string]$m) { Write-Host "`n$m" -ForegroundColor Cyan }

# ------------------------------------------------------------------ .env

function Read-EnvFile ([string]$Path) {
    if (-not (Test-Path $Path)) {
        throw "env file not found: $Path"
    }
    $map = @{}
    foreach ($line in Get-Content $Path) {
        $t = $line.Trim()
        if ($t -eq '' -or $t.StartsWith('#')) { continue }
        $i = $t.IndexOf('=')
        if ($i -lt 1) { continue }
        $key = $t.Substring(0, $i).Trim()
        $val = $t.Substring($i + 1)
        # strip a trailing inline comment ( whitespace then # ), which .env uses heavily
        $val = [regex]::Replace($val, '\s+#.*$', '')
        $map[$key] = $val.Trim().Trim('"').Trim("'")
    }
    return $map
}

function Get-Val ($Map, [string]$Key, [string]$Default = '') {
    if ($Map.ContainsKey($Key) -and $Map[$Key] -ne '') { return $Map[$Key] }
    return $Default
}

# ------------------------------------------------------------------ Gemini

function Test-Gemini ($Env) {
    Write-Head 'GEMINI'
    $key = Get-Val $Env 'GEMINI_API_KEY'
    if ($key -eq '') {
        Write-Bad 'GEMINI_API_KEY is empty'
        Write-Fix 'get one at https://aistudio.google.com/apikey and put it in .env'
        return $false
    }
    Write-Ok ("GEMINI_API_KEY present ({0} chars, starts {1}...)" -f $key.Length, $key.Substring(0, [Math]::Min(4, $key.Length)))
    # Deliberately NOT asserting a prefix. Older AI Studio keys start "AIza", newer ones
    # start "AQ."; hard-coding either turns a working key into a scary red FAIL. The live
    # call below is the only check that actually means anything.
    if ($key.Length -lt 30) {
        Write-Bad ("that is only {0} characters, which is short for an API key" -f $key.Length)
        Write-Fix 'you may have copied a project id or truncated the paste'
    }

    $model = Get-Val $Env 'GEMINI_MODEL' 'gemini-3.5-flash-lite'
    Write-Ok "GEMINI_MODEL = $model"

    $uri  = "https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent"
    $body = @{ contents = @(@{ parts = @(@{ text = 'Reply with exactly the word: pong' }) }) } | ConvertTo-Json -Depth 8

    try {
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $resp = Invoke-RestMethod -Method Post -Uri $uri `
            -Headers @{ 'x-goog-api-key' = $key } `
            -ContentType 'application/json' -Body $body -TimeoutSec 30
        $sw.Stop()
    }
    catch {
        $detail = ''
        $status = ''
        # PS 5.1 hides the response body on a non-2xx; dig it out, it is where the reason is.
        try {
            $r = $_.Exception.Response
            if ($r) {
                $status = [int]$r.StatusCode
                $sr = New-Object System.IO.StreamReader($r.GetResponseStream())
                $detail = $sr.ReadToEnd()
                $sr.Close()
            }
        } catch { }
        if ($detail -eq '') { $detail = $_.Exception.Message }

        Write-Bad ("the call failed (HTTP {0}): {1}" -f $status, $detail.Substring(0, [Math]::Min(300, $detail.Length)))
        $low = $detail.ToLower()
        if ($low -match 'api key not valid|api_key_invalid') {
            Write-Fix 'the key is wrong or was revoked -- make a new one at https://aistudio.google.com/apikey'
        }
        elseif ($status -eq 429 -or $low -match 'resource_exhausted|quota') {
            Write-Fix 'rate limited. See https://ai.google.dev/gemini-api/docs/rate-limits and raise VISION_MIN_INTERVAL_S'
        }
        elseif ($status -eq 404 -or $low -match 'not found') {
            Write-Fix "the model '$model' is not available to this key. See https://ai.google.dev/gemini-api/docs/models and set GEMINI_MODEL"
        }
        elseif ($status -eq 403 -or $low -match 'permission') {
            Write-Fix 'the key exists but is not enabled for the Gemini API -- regenerate it from AI Studio, not a Cloud console project'
        }
        return $false
    }

    $text = ''
    try { $text = $resp.candidates[0].content.parts[0].text.Trim() } catch { }
    Write-Ok ("live call succeeded in {0:N1}s, model replied: '{1}'" -f $sw.Elapsed.TotalSeconds, $text)

    $interval = Get-Val $Env 'VISION_MIN_INTERVAL_S'
    if ($interval -ne '') {
        $rpm = [Math]::Round(60.0 / [double]$interval)
        Write-Ok "VISION_MIN_INTERVAL_S = $interval -> at most $rpm calls/min. Confirm your tier allows that at https://ai.google.dev/gemini-api/docs/rate-limits"
    }
    return $true
}

# ------------------------------------------------------------------ Gmail

function Test-Smtp ($Env) {
    Write-Head 'GMAIL / SMTP'
    $user = Get-Val $Env 'SMTP_USER'
    $pass = Get-Val $Env 'SMTP_APP_PASSWORD'
    $to   = Get-Val $Env 'ALERT_TO'
    $host_ = Get-Val $Env 'SMTP_HOST' 'smtp.gmail.com'
    $port  = [int](Get-Val $Env 'SMTP_PORT' '587')

    $missing = @()
    if ($user -eq '') { $missing += 'SMTP_USER' }
    if ($pass -eq '') { $missing += 'SMTP_APP_PASSWORD' }
    if ($to   -eq '') { $missing += 'ALERT_TO' }
    if ($missing.Count -gt 0) {
        Write-Bad ("empty in .env: {0}" -f ($missing -join ', '))
        Write-Fix 'app password: https://myaccount.google.com/apppasswords (needs 2-Step Verification on first)'
        return $false
    }

    # Checked before handing them to .NET: MailAddress throws a MethodInvocationException
    # that says nothing useful about WHICH address was malformed. A missing @ is a typo
    # people make constantly and it deserves a one-line answer, not a stack trace.
    $badAddr = @()
    foreach ($pair in @(@('SMTP_USER', $user), @('ALERT_TO', $to))) {
        if ($pair[1] -notmatch '^[^@\s]+@[^@\s]+\.[^@\s]+$') { $badAddr += "$($pair[0]) = $($pair[1])" }
    }
    if ($badAddr.Count -gt 0) {
        Write-Bad ("not a valid email address: {0}" -f ($badAddr -join '; '))
        Write-Fix 'check for a missing @ or a typo in the domain'
        return $false
    }

    Write-Ok "SMTP_USER = $user"
    Write-Ok "ALERT_TO  = $to"
    Write-Ok ("SMTP_APP_PASSWORD present ({0} chars)" -f $pass.Length)
    if ($pass.Contains(' ')) {
        Write-Bad 'the app password contains spaces'
        Write-Fix 'Google displays it as 4 groups of 4 -- paste it as 16 characters, no spaces'
        return $false
    }
    if ($pass.Length -ne 16) {
        Write-Bad ("app passwords are 16 characters, this is {0}" -f $pass.Length)
        Write-Fix 'you may have pasted your account password instead'
    }
    Write-Ok "server = ${host_}:${port}"

    $client = $null
    $msg    = $null
    try {
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $client = New-Object System.Net.Mail.SmtpClient($host_, $port)
        $client.EnableSsl   = $true            # STARTTLS on 587
        $client.Timeout     = 20000
        $client.Credentials = New-Object System.Net.NetworkCredential($user, $pass)

        $msg = New-Object System.Net.Mail.MailMessage
        $msg.From = New-Object System.Net.Mail.MailAddress($user)
        $msg.To.Add($to)
        $msg.Subject = '[relay] credential check'
        $msg.Body    = 'If you are reading this, Project Relay can send mail. Sent by scripts/Verify-Credentials.ps1.'

        $client.Send($msg)
        $sw.Stop()
    }
    catch [System.Net.Mail.SmtpException] {
        $m = $_.Exception.Message
        Write-Bad "send failed: $m"
        if ($m -match '5\.7\.8|not accepted|authentication') {
            Write-Fix '535 here almost always means a normal account password was used, or 2-Step Verification is off.'
            Write-Fix 'Make an APP password at https://myaccount.google.com/apppasswords'
        }
        elseif ($m -match 'timed out|timeout') {
            Write-Fix "a firewall is probably blocking outbound port $port"
        }
        return $false
    }
    catch {
        Write-Bad ("send failed: {0}: {1}" -f $_.Exception.GetType().Name, $_.Exception.Message)
        return $false
    }
    finally {
        if ($msg)    { $msg.Dispose() }
        if ($client) { $client.Dispose() }
    }

    Write-Ok ("sent in {0:N1}s -- check the inbox of {1}" -f $sw.Elapsed.TotalSeconds, $to)
    return $true
}

# ------------------------------------------------------------------ main

if (-not $EnvFile) {
    $EnvFile = Join-Path (Split-Path -Parent $PSScriptRoot) '.env'
}

try {
    $envMap = Read-EnvFile $EnvFile
}
catch {
    Write-Bad $_.Exception.Message
    exit 2
}
Write-Host "loaded $EnvFile"

$both    = -not ($Gemini -or $Smtp)
$results = @()

if ($both -or $Gemini) { $results += [pscustomobject]@{ Name = 'gemini'; Pass = (Test-Gemini $envMap) } }
if ($both -or $Smtp)   { $results += [pscustomobject]@{ Name = 'smtp';   Pass = (Test-Smtp   $envMap) } }

Write-Host ("`n" + ('-' * 52))
$allPass = $true
foreach ($r in $results) {
    if ($r.Pass) {
        Write-Host ("  {0,-8} PASS" -f $r.Name) -ForegroundColor Green
    }
    else {
        Write-Host ("  {0,-8} FAIL" -f $r.Name) -ForegroundColor Red
        $allPass = $false
    }
}

if ($allPass) { exit 0 } else { exit 1 }
