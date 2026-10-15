param(
    [string]$Serial = "17513b4",
    [string]$Adb = "D:\PhD_LetGoo\PhD_Farming\edge-ai\.tools\platform-tools\adb.exe",
    [string]$Revision = "5359861c739e955e79d9a303bcbc70fb988958b1"
)

$ErrorActionPreference = "Stop"
$repo = "ggerganov/whisper.cpp"
$file = "ggml-base-q8_0.bin"
$expectedSha256 = "C577B9A86E7E048A0B7EADA054F4DD79A56BBFA911FBDACF900AC5B567CBB7D9"
$stage = Join-Path $env:TEMP "omniglass-whisper-base-q8"
$localModel = Join-Path $stage $file
$remoteDir = "/data/local/tmp/aibox-eye/models/whisper"

if (-not (Test-Path -LiteralPath $Adb)) { throw "ADB not found: $Adb" }
if (-not (Get-Command hf -ErrorAction SilentlyContinue)) { throw "Hugging Face CLI 'hf' is required" }

New-Item -ItemType Directory -Force -Path $stage | Out-Null
& hf download $repo $file --revision $Revision --local-dir $stage
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $localModel)) {
    throw "Failed to download $repo/$file"
}
$actualSha256 = (Get-FileHash -LiteralPath $localModel -Algorithm SHA256).Hash
if ($actualSha256 -ne $expectedSha256) {
    throw "Model checksum mismatch: expected $expectedSha256, got $actualSha256"
}

& $Adb -s $Serial shell "mkdir -p $remoteDir"
& $Adb -s $Serial push $localModel "$remoteDir/$file"
if ($LASTEXITCODE -ne 0) { throw "Failed to install $file on $Serial" }
& $Adb -s $Serial shell "test -s $remoteDir/$file"
if ($LASTEXITCODE -ne 0) { throw "Installed model is missing or empty" }

Write-Output "Installed pinned Whisper Base Q8: $remoteDir/$file"
Write-Output "SHA256: $actualSha256"
