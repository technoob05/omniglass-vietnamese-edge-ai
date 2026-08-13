<#
Starts the persistent MiniCPM-V 4.6 vision/text service on a rooted QCS8550
box. The model stays loaded between requests. It is intentionally a
single-flight, turn-based service; it is not the MiniCPM-o full-duplex server.
#>
[CmdletBinding()]
param(
    [string]$Serial = '17513b4',
    [string]$Adb = 'D:\PhD_LetGoo\PhD_Farming\edge-ai\.tools\platform-tools\adb.exe',
    [int]$Port = 18191
)

$ErrorActionPreference = 'Stop'
$runtime = '/data/local/tmp/omniglass-v46-resident-v1/runtime'
$modelRoot = '/data/local/tmp/omniglass-minicpm-v46-v1/models'
$log = '/data/local/tmp/omniglass-v46-resident-v1/server.log'
$command = @"
set -eu
test "`$(id -u)" = 0
test -x $runtime/bin/llama-mtmd-resident-server
test -f $modelRoot/MiniCPM-V-4_6-Q4_0.gguf
test -f $modelRoot/mmproj-model-f16.gguf
pkill -f 'llama-mtmd-resident-server' || true
export LD_LIBRARY_PATH=$runtime/lib
export ADSP_LIBRARY_PATH=$runtime/lib
export MTMD_BACKEND_DEVICE=HTP0
RESIDENT_VLM_PORT=$Port nohup $runtime/bin/llama-mtmd-resident-server \
  -m $modelRoot/MiniCPM-V-4_6-Q4_0.gguf \
  --mmproj $modelRoot/mmproj-model-f16.gguf \
  -c 2048 -n 32 --image-min-tokens 64 --image-max-tokens 64 \
  --device HTP0 -ngl 99 --mmproj-offload --no-mmap >$log 2>&1 &
"@

& $Adb -s $Serial root | Out-Null
& $Adb -s $Serial wait-for-device
if ((& $Adb -s $Serial shell id) -notmatch 'uid=0') {
    throw 'ADB root is required: HTP/FastRPC device access is unavailable to the shell user.'
}
& $Adb -s $Serial shell "mkdir -p /data/local/tmp/omniglass-v46-resident-v1/inbox"
& $Adb -s $Serial shell $command
& $Adb -s $Serial forward "tcp:$Port" "tcp:$Port" | Out-Null

for ($i = 0; $i -lt 30; $i++) {
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2
        if ($health.status -eq 'ok') {
            Write-Output "MiniCPM-V resident service ready at http://127.0.0.1:$Port (HTP0 vision)."
            exit 0
        }
    } catch { Start-Sleep -Seconds 1 }
}
& $Adb -s $Serial shell "tail -n 120 $log"
throw 'Resident service did not become healthy.'
