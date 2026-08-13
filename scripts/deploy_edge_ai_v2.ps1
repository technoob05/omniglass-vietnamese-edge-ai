param(
    [string]$Serial = "17513b4",
    [string]$Adb = "D:\PhD_LetGoo\PhD_Farming\edge-ai\.tools\platform-tools\adb.exe"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$config = Join-Path $root "versions\edge-ai-v2\device\config\qwen35-production.json"
$vlm = Join-Path $root "versions\edge-ai-v2\device\aibox_eye\vlm.py"
$server = Join-Path $root "versions\edge-ai-v2\device\aibox_eye\server.py"
$remoteRoot = "/data/local/tmp/aibox-eye"
$remoteConfig = "$remoteRoot/config/production.json"
$backup = "$remoteRoot/version-1-production.json"

if (-not (Test-Path -LiteralPath $Adb)) { throw "ADB not found: $Adb" }
if (-not (Test-Path -LiteralPath $config)) { throw "Missing v2 config: $config" }
if (-not (Test-Path -LiteralPath $vlm)) { throw "Missing v2 VLM client: $vlm" }
if (-not (Test-Path -LiteralPath $server)) { throw "Missing v2 web server: $server" }

& $Adb -s $Serial get-state | Out-Null
if ($LASTEXITCODE -ne 0) { throw "ADB device is not ready: $Serial" }

& $Adb -s $Serial shell "test -f $backup || cp $remoteConfig $backup"
& $Adb -s $Serial push $config $remoteConfig | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Failed to push v2 config" }
& $Adb -s $Serial push $vlm "$remoteRoot/aibox_eye/vlm.py" | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Failed to push v2 VLM client" }
& $Adb -s $Serial push $server "$remoteRoot/aibox_eye/server.py" | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Failed to push v2 web server" }

# Restart only the coordinator so the new profile is loaded. QNN perception
# and GenieX remain untouched.
& $Adb -s $Serial shell "sh $remoteRoot/stop_on_aibox.sh"
& $Adb -s $Serial shell "sh $remoteRoot/start_on_aibox.sh"
if ($LASTEXITCODE -ne 0) { throw "AIBOX-eye v2 did not pass startup health checks" }

Write-Output "Edge AI v2 deployed: $remoteConfig"
Write-Output "Backup preserved at: $backup"
Write-Output "VLM model: local/Qwen3.5-2B-GGUF:Q4_0"
