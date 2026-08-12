param(
    [int]$LocalPort = 7873,
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,
    [int]$SshPort = 22,
    [string]$RemoteUser = $env:USERNAME
)

$ErrorActionPreference = "SilentlyContinue"
$sshTarget = "$RemoteUser@$RemoteHost"
$forward = "${LocalPort}:127.0.0.1:${LocalPort}"

while ($true) {
    $listener = Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue
    if (-not $listener) {
        $arguments = @(
            "-N",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "TCPKeepAlive=yes",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=6",
            "-o", "StrictHostKeyChecking=yes",
            "-L", $forward,
            "-p", $SshPort,
            $sshTarget
        )
        $tunnel = Start-Process -FilePath "ssh.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru
        if ($tunnel) {
            Wait-Process -Id $tunnel.Id
        }
    }
    Start-Sleep -Seconds 2
}
