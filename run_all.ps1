$ErrorActionPreference = "Continue"
$root = "E:\3dgs_ladia"
Set-Location $root
$env:PYTHONUNBUFFERED = "1"
$env:PATH = "E:\gaussian-splatting-windows\envs\gaussian_splatting;E:\gaussian-splatting-windows\envs\gaussian_splatting\Library\bin;E:\gaussian-splatting-windows\envs\gaussian_splatting\Lib\site-packages\torch\lib;" + $env:PATH
$env:PYTHONPATH = "E:\3dgs_ladia\submodules\diff-gaussian-rasterization\build\lib.win-amd64-cpython-37"
$py = "E:\gaussian-splatting-windows\envs\gaussian_splatting\python.exe"
$scenes = @("courtyard", "delivery", "electro", "kicker", "meadow", "office", "pipes")
New-Item -ItemType Directory -Force -Path "$root\output\logs" | Out-Null
$status = "$root\output\STATUS.txt"

function Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -LiteralPath $status -Value $line -Encoding UTF8
    Write-Host $line
}

function VertexCount($ply) {
    $fs = [System.IO.File]::OpenRead($ply)
    $buf = New-Object byte[] 512
    $n = $fs.Read($buf, 0, 512)
    $fs.Close()
    $text = [System.Text.Encoding]::ASCII.GetString($buf, 0, $n)
    $m = [regex]::Match($text, "element vertex (\d+)")
    if ($m.Success) { return $m.Groups[1].Value }
    return "unknown"
}

function TrainOne($scene, $logName, $extra) {
    $src = "$root\data\prepared\${scene}_lidar"
    $dst = "$root\output\${scene}_lidar"
    $log = "$root\output\logs\$logName"
    $trainArgs = @(
        "$root\train.py",
        "-s", $src,
        "-m", $dst,
        "--disable_viewer",
        "--antialiasing",
        "--train_test_exp",
        "--exposure_lr_init", "0.001",
        "--exposure_lr_final", "0.0001",
        "--exposure_lr_delay_steps", "5000",
        "--exposure_lr_delay_mult", "0.001"
    ) + $extra
    & $py @trainArgs *> $log
    return $LASTEXITCODE
}

Log "queue start: $($scenes -join ', ')"
foreach ($scene in $scenes) {
    $prepared = "$root\data\prepared\${scene}_lidar\sparse\0\points3D.ply"
    if (-not (Test-Path -LiteralPath $prepared)) {
        Log "SKIP $scene missing $prepared"
        continue
    }
    Log "TRAIN $scene"
    $code = TrainOne $scene "$scene.log" @()
    $ply = "$root\output\${scene}_lidar\point_cloud\iteration_30000\point_cloud.ply"
    if ($code -ne 0 -or -not (Test-Path -LiteralPath $ply)) {
        $oom = Select-String -LiteralPath "$root\output\logs\$scene.log" -Pattern "out of memory" -Quiet
        if ($oom) {
            Log "OOM $scene, retry with densify_grad_threshold 0.0005"
            if (Test-Path -LiteralPath "$root\output\${scene}_lidar") {
                Remove-Item -LiteralPath "$root\output\${scene}_lidar" -Recurse -Force
            }
            $code = TrainOne $scene "${scene}_retry.log" @("--densify_grad_threshold", "0.0005")
        }
    }
    if (Test-Path -LiteralPath $ply) {
        $n = VertexCount $ply
        Log "DONE $scene exit=$code gaussians=$n ply=$ply"
    } else {
        Log "FAIL $scene exit=$code see output\logs\$scene.log"
    }
}
Log "ALL FINISHED"
