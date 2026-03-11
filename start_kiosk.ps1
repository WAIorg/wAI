$url = "http://localhost:5174/"

$edgePaths = @(
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe"
)

$edge = $edgePaths | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $edge) {
    Write-Host "Edge not found."
    exit
}

Start-Process -FilePath $edge -ArgumentList @(
    "--kiosk",
    $url,
    "--edge-kiosk-type=fullscreen",
    "--no-first-run",
    "--force-device-scale-factor=0.5"
)