<#
.SYNOPSIS
    Assemble the Windows client bundle around a PyInstaller build.

.DESCRIPTION
    Produces the folder the client actually receives:

        Pose3D-Windows\
          Pose3D.exe + _internal\    the app
          blender\blender.exe        the export engine
          models\*.onnx              pose weights, so nothing downloads
          workspace\                 projects and exports default here
          README.txt                 how to run it, incl. SmartScreen

    Blender and the weights sit BESIDE the exe rather than inside it because
    pose3d.runtime.app_dir() resolves them relative to the executable, and
    because burying a 1 GB Blender inside a onefile build would unpack it on
    every launch.

    Run after `pyinstaller pose3d.spec`. Downloads are cached in .cache\ so
    re-runs and CI cache hits are cheap.

.EXAMPLE
    pwsh tools/make_windows_bundle.ps1 -Dist dist/Pose3D -Out build/Pose3D-Windows
#>
[CmdletBinding()]
param(
    [string]$Dist = "dist/Pose3D",
    [string]$Out = "build/Pose3D-Windows",
    [string]$BlenderVersion = "5.1.1",
    [string]$CacheDir = ".cache",
    [switch]$SkipBlender,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    if (-not (Test-Path $Dist)) {
        throw "$Dist not found. Run: pyinstaller pose3d.spec --noconfirm --clean"
    }
    if (-not (Test-Path (Join-Path $Dist "Pose3D.exe"))) {
        throw "$Dist has no Pose3D.exe — check the spec's EXE(name=...)."
    }

    if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

    Write-Host "==> app"
    Copy-Item -Recurse -Force (Join-Path $Dist "*") $Out

    # --- Blender ---------------------------------------------------------
    if (-not $SkipBlender) {
        $short = ($BlenderVersion -split '\.')[0..1] -join '.'
        $zipName = "blender-$BlenderVersion-windows-x64.zip"
        $archive = Join-Path $CacheDir $zipName
        if (-not (Test-Path $archive)) {
            $url = "https://download.blender.org/release/Blender$short/$zipName"
            Write-Host "==> downloading $url"
            # ~414 MB; the progress bar makes Invoke-WebRequest crawl
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $url -OutFile $archive -UseBasicParsing
        } else {
            Write-Host "==> using cached $archive"
        }

        Write-Host "==> blender"
        $stage = Join-Path $CacheDir "blender-stage"
        if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
        Expand-Archive -Path $archive -DestinationPath $stage -Force
        $inner = Get-ChildItem $stage -Directory | Select-Object -First 1
        Move-Item $inner.FullName (Join-Path $Out "blender")
        Remove-Item -Recurse -Force $stage

        # Translations are ~100 MB of a bundle nobody will read in Klingon.
        $locale = Join-Path $Out "blender\*\datafiles\locale"
        Get-Item $locale -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

        $exe = Join-Path $Out "blender\blender.exe"
        if (-not (Test-Path $exe)) { throw "no blender.exe under $Out\blender" }
    }

    # --- pose weights ----------------------------------------------------
    # Staged into the cache first, then copied in, so a CI cache of $CacheDir
    # actually spares the ~150 MB download. Fetching straight into the bundle
    # would re-download on every run.
    Write-Host "==> weights"
    $cacheModels = Join-Path $CacheDir "models"
    & python tools/fetch_weights.py --out $cacheModels
    if ($LASTEXITCODE -ne 0) { throw "fetch_weights.py failed" }
    $models = Join-Path $Out "models"
    New-Item -ItemType Directory -Force -Path $models | Out-Null
    Copy-Item -Force (Join-Path $cacheModels "*.onnx") $models

    # --- workspace + readme ----------------------------------------------
    New-Item -ItemType Directory -Force -Path (Join-Path $Out "workspace") | Out-Null
    Copy-Item -Force "packaging/windows/README.txt" (Join-Path $Out "README.txt")

    $mb = [math]::Round(
        ((Get-ChildItem $Out -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 0)
    Write-Host "==> $Out assembled, $mb MB"

    if ($Zip) {
        $archiveOut = "$Out.zip"
        if (Test-Path $archiveOut) { Remove-Item -Force $archiveOut }
        Write-Host "==> zipping to $archiveOut"
        # 7-Zip where available: Compress-Archive takes many minutes on a 1.6 GB
        # tree and is uncomfortably close to its 2 GB ceiling.
        if (Get-Command 7z -ErrorAction SilentlyContinue) {
            & 7z a -tzip -mx=5 -bso0 -bsp0 $archiveOut $Out | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "7z failed ($LASTEXITCODE)" }
        } else {
            $ProgressPreference = 'SilentlyContinue'
            Compress-Archive -Path $Out -DestinationPath $archiveOut -CompressionLevel Optimal
        }
        $zmb = [math]::Round((Get-Item $archiveOut).Length / 1MB, 0)
        Write-Host "==> $archiveOut, $zmb MB"
    }
}
finally {
    Pop-Location
}
