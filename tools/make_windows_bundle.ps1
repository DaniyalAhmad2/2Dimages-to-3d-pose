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
          Diagnose.cmd               self-test + log, for a support email

    Blender and the weights sit BESIDE the exe rather than inside it because
    pose3d.runtime.app_dir() resolves them relative to the executable, and
    because burying a 1 GB Blender inside a onefile build would unpack it on
    every launch.

    Which Blender, from where, and with which checksum comes out of
    packaging/windows/inputs.json via tools/blender_input.py. Nothing here
    writes a version down: it used to, in a parameter default, while both
    workflows wrote the same version again in a URL of their own.

    Run after `pyinstaller pose3d.spec`. Downloads are cached in .cache\ so
    re-runs and CI cache hits are cheap.

.PARAMETER Python
    The interpreter to run the repo's tools with. Pass the venv's
    sys.executable. `python` off PATH is whatever the machine happens to
    resolve — on a Windows runner, often the Microsoft Store stub, which is
    not the environment the lockfile installed.

.EXAMPLE
    pwsh tools/make_windows_bundle.ps1 -Dist dist/Pose3D -Out build/Pose3D-Windows
#>
[CmdletBinding()]
param(
    [string]$Dist = "dist/Pose3D",
    [string]$Out = "build/Pose3D-Windows",
    [string]$Python = "python",
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

    # Every tool below is run through this, and the failure worth catching is
    # not "no python" — it is the WRONG python: the Store stub, or a system
    # interpreter that is not the environment the lockfile installed. Naming
    # the two imports those tools need turns that into one message here
    # instead of an ImportError three steps in.
    & $Python -c "import pefile, rtmlib, sys; print(sys.executable)"
    if ($LASTEXITCODE -ne 0) {
        throw ("-Python '$Python' cannot run this repo's tools. Pass the " +
               "venv's interpreter (its sys.executable).")
    }

    if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

    Write-Host "==> app"
    Copy-Item -Recurse -Force (Join-Path $Dist "*") $Out

    # Before anything else lands here: every DLL the app imports must be in the
    # bundle or part of Windows itself. A build machine has the Visual C++
    # runtime and Qt's dependencies installed system-wide and so cannot notice
    # one missing from _internal\ — the client's machine notices, with
    # "Failed to load Python DLL ... The specified module could not be found."
    # Run now, while $Out holds only the app: Blender is a separate release
    # with its own C runtime and is not ours to audit.
    Write-Host "==> dependency audit"
    & $Python tools/check_bundle_deps.py $Out
    if ($LASTEXITCODE -ne 0) { throw "check_bundle_deps.py failed" }

    # --- Blender ---------------------------------------------------------
    if (-not $SkipBlender) {
        function Get-BlenderInput([string]$Field) {
            $value = & $Python tools/blender_input.py $Field
            if ($LASTEXITCODE -ne 0) { throw "blender_input.py $Field failed" }
            return $value.Trim()
        }
        $zipName = Get-BlenderInput "--zip-name"
        # Refuses when the checksum is still the FILL-FROM-CI placeholder:
        # a build that cannot verify its 414 MB download must stop, not warn.
        $wantHash = Get-BlenderInput "--sha256"
        $archive = Join-Path $CacheDir $zipName

        if (-not (Test-Path $archive)) {
            $url = Get-BlenderInput "--url"
            Write-Host "==> downloading $url"
            # ~414 MB; the progress bar makes Invoke-WebRequest crawl
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $url -OutFile $archive -UseBasicParsing
        } else {
            Write-Host "==> using cached $archive"
        }

        # Both paths, not just the download: a cached archive is a file some
        # earlier run left behind, and a run interrupted mid-download leaves a
        # short one that unzips into a blender\ folder missing whatever came
        # after the cut. Deleted on mismatch so the next run cannot "use
        # cached" the same bad file forever.
        Write-Host "==> verifying $zipName"
        $gotHash = (Get-FileHash -Algorithm SHA256 -Path $archive).Hash.ToLower()
        if ($gotHash -ne $wantHash) {
            Remove-Item -Force $archive
            throw ("checksum mismatch for ${zipName}:`n" +
                   "  expected $wantHash  (packaging/windows/inputs.json)`n" +
                   "  actual   $gotHash`n" +
                   "The archive has been deleted; re-run to fetch it again.")
        }

        Write-Host "==> blender"
        $stage = Join-Path $CacheDir "blender-stage"
        if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
        Expand-Archive -Path $archive -DestinationPath $stage -Force
        $inner = Get-ChildItem $stage -Directory | Select-Object -First 1
        Move-Item $inner.FullName (Join-Path $Out "blender")
        Remove-Item -Recurse -Force $stage

        # Translations are ~100 MB of a bundle nobody will read in Klingon.
        # It used to be -ErrorAction SilentlyContinue on a wildcard path: when
        # a Blender release moves that directory, the purge deletes nothing,
        # says nothing, and the bundle silently grows back by 100 MB.
        # Test-Path first: a wildcard that matches nothing makes Get-Item
        # raise its own error, and this failure deserves to say what it means.
        $locale = Join-Path $Out "blender\*\datafiles\locale"
        if (-not (Test-Path $locale)) {
            throw ("no locale directory at ${locale}: a Blender release has " +
                   "moved it, and the bundle would quietly grow 100 MB.")
        }
        $purged = @(Get-ChildItem $locale -Recurse -File)
        $purgedMb = [math]::Round(
            (($purged | Measure-Object Length -Sum).Sum / 1MB), 0)
        Remove-Item -Recurse -Force $locale
        Write-Host "==> locale purged: $($purged.Count) files, $purgedMb MB"

        $exe = Join-Path $Out "blender\blender.exe"
        if (-not (Test-Path $exe)) { throw "no blender.exe under $Out\blender" }
    }

    # --- pose weights ----------------------------------------------------
    # Staged into the cache first, then copied in, so a CI cache of $CacheDir
    # actually spares the ~150 MB download. Fetching straight into the bundle
    # would re-download on every run. fetch_weights.py hashes everything it
    # staged, including files it found already there, against inputs.json.
    Write-Host "==> weights"
    $cacheModels = Join-Path $CacheDir "models"
    & $Python tools/fetch_weights.py --out $cacheModels
    if ($LASTEXITCODE -ne 0) { throw "fetch_weights.py failed" }
    $models = Join-Path $Out "models"
    New-Item -ItemType Directory -Force -Path $models | Out-Null
    Copy-Item -Force (Join-Path $cacheModels "*.onnx") $models

    # --- workspace + readme ----------------------------------------------
    New-Item -ItemType Directory -Force -Path (Join-Path $Out "workspace") | Out-Null
    Copy-Item -Force "packaging/windows/README.txt" (Join-Path $Out "README.txt")
    # The fallback for a diagnose exe that will not run: a double-clickable
    # script that runs the self-test and shows the log. It costs 1 KB and it
    # is the only route to a diagnosis that needs neither the GUI nor the
    # second executable.
    Copy-Item -Force "packaging/windows/Diagnose.cmd" (Join-Path $Out "Diagnose.cmd")

    $mb = [math]::Round(
        ((Get-ChildItem $Out -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 0)
    Write-Host "==> $Out assembled, $mb MB"

    if ($Zip) {
        # tools/bundle_zip.py, not a block of PowerShell: the MAX_PATH gate
        # that keeps the client's "path too long" from coming back is a rule
        # about entry names, it is tested on Linux, and it must be the same
        # rule here as in the release workflow. The one that used to live in
        # this file had no gate at all and no CI job ever ran it.
        Write-Host "==> zipping"
        & $Python tools/bundle_zip.py $Out
        if ($LASTEXITCODE -ne 0) { throw "bundle_zip.py failed" }
    }
}
finally {
    Pop-Location
}
