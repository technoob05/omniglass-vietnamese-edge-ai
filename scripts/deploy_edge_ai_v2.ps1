param(
    [string]$Serial = "17513b4",
    [string]$Adb = "D:\PhD_LetGoo\PhD_Farming\edge-ai\.tools\platform-tools\adb.exe"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$config = Join-Path $root "versions\edge-ai-v2\device\config\qwen35-production.json"
$runtimeFiles = @("answers.py", "config.py", "intents.py", "keyframes.py", "orchestrator.py", "server.py", "tts.py", "vlm.py")
$runtimeRoot = Join-Path $root "versions\edge-ai-v2\device\aibox_eye"
$remoteRoot = "/data/local/tmp/aibox-eye"
$remoteConfig = "$remoteRoot/config/production.json"
$backup = "$remoteRoot/version-1-production.json"

if (-not (Test-Path -LiteralPath $Adb)) { throw "ADB not found: $Adb" }
if (-not (Test-Path -LiteralPath $config)) { throw "Missing v2 config: $config" }
foreach ($name in $runtimeFiles) {
    $source = Join-Path $runtimeRoot $name
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing v2 runtime file: $source" }
}

& $Adb -s $Serial get-state | Out-Null
if ($LASTEXITCODE -ne 0) { throw "ADB device is not ready: $Serial" }

& $Adb -s $Serial shell "test -f $backup || cp $remoteConfig $backup"
& $Adb -s $Serial push $config $remoteConfig | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Failed to push v2 config" }
foreach ($name in $runtimeFiles) {
    $source = Join-Path $runtimeRoot $name
    & $Adb -s $Serial push $source "$remoteRoot/aibox_eye/$name" | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Failed to push v2 runtime file: $name" }
}

# Restart only the coordinator so the new profile is loaded. QNN perception
# and GenieX remain untouched.
& $Adb -s $Serial shell "sh $remoteRoot/stop_on_aibox.sh"
& $Adb -s $Serial shell "sh $remoteRoot/start_on_aibox.sh"
if ($LASTEXITCODE -ne 0) { throw "AIBOX-eye v2 did not pass startup health checks" }

Write-Output "Edge AI v2 deployed: $remoteConfig"
Write-Output "Backup preserved at: $backup"
Write-Output "VLM model: local/Qwen3.5-2B-GGUF:Q4_0"
