# The Windows release gate

Every Windows fault this project has shipped was invisible to CI, because CI
tested something *adjacent* to what the client received.

| what the client did | what CI did |
|---|---|
| extracted `Pose3D-Windows.zip` | ran `build\Pose3D-Windows\`, straight out of PyInstaller |
| into `C:\Users\<name>\Downloads\…` | onto `D:\a\pose3d-tool\pose3d-tool\build`, no spaces |
| on a machine with no Visual C++ runtime | on a runner with the runtime, and most of Qt, installed system-wide |
| double-clicked a windowed exe | read a console the windowed exe never had |

So the gate now **mirrors the client**: it builds the zip, extracts it with
7-Zip into a nested path with a space in it, and runs the self-test out of
*that* copy with Blender and the weights cache deliberately unavailable.

Everything below lives in one composite action,
`.github/actions/windows-bundle/action.yml`, used verbatim by two workflows:

* `.github/workflows/windows-bundle.yml` — the gate. Runs on `push: main`, on
  pull requests that touch anything able to break a bundle, and on demand.
  90 minutes.
* `.github/workflows/windows-release.yml` — the release. The same action, then
  `upload-artifact` and the GitHub Release. Runs on `v*` tags and on demand.

They share the action so the archive attached to a release is one the gate has
already extracted and self-tested. A release built by steps of its own would
drift from the gate, and the copy that ships would be the one nobody ran.

`tests/test_release_gate.py` asserts the mirror is still a mirror — it reads
these files, because there is no second Windows machine here to run them on.

## The gate, in order

| # | check | the failure it buys |
|---|---|---|
| 1 | read `.python-version`, then `setup-uv` with `python-version:` that value | `uv.lock` says only `>=3.12`, so an unpinned runner freezes the exe against whatever interpreter it has newest — while every doc, README string and test in this repo names `python312.dll`. Read in a step of its own because setup-uv has no `python-version-file` input (not on v5, not on `main`) and an unknown input is *silently ignored*: passing one would have pinned nothing and said nothing. A missing or empty `.python-version` fails the job. |
| 2 | `uv sync --frozen --all-groups` | resolving here would ship the client versions nothing was tested against. |
| 3 | cache `.cache`, keyed on `hashFiles('packaging/windows/inputs.json')` | `inputs.json` is the only place a Blender version, URL or checksum is written. Keying the cache on anything else means a corrected checksum keeps restoring the old bytes and every build fails on a file nobody can see. |
| 4 | `pyinstaller pose3d.spec --noconfirm --clean` | — (the build itself) |
| 5 | `make_windows_bundle.ps1 -Python <venv sys.executable>` | `python` off PATH on a Windows runner is often the Microsoft Store stub, not the environment the lockfile installed. Inside the script: the Blender download is verified against its published sha256 (a truncated 414 MB archive unzips into a `blender\` folder that fails on the client, not here), the locale purge `throw`s rather than silently leaving 100 MB, and the weights are hashed against `inputs.json`. |
| 6 | …and, inside the same script, `check_bundle_deps.py` — **once** | a DLL missing from `_internal\` still loads on this runner, which has the Visual C++ runtime installed system-wide, and only fails on a clean machine: *"Failed to load Python DLL … python312.dll. LoadLibrary: The specified module could not be found."* Run while the bundle holds only the app, before Blender lands beside it — Blender ships its own C runtime and is not ours to audit. The release workflow used to run the audit a second time for the same answer. |
| 7 | `check_bundle_layout.py build/Pose3D-Windows` | the other question: is everything *there at all*? Qt's `platforms\qwindows.dll`, `imageformats\qjpeg.dll`, the styles plugin, `opengl32sw.dll`, `onnxruntime\*.dll`, `character.blend`, `dark.qss`, `blender_job.py`, three `models\*.onnx` hashed against `inputs.json`, `blender\blender.exe`. Nothing asked this before. Run before zipping, so a missing file is named now rather than after five minutes of compression. |
| 8 | `bundle_zip.py --limit 180` | the first bundle the client received would not extract: *"path too long"*. Archived from the bundle's parent so entries begin at `Pose3D-Windows\` — six characters that come straight off their extraction path — and refused if any entry exceeds 180 characters, which leaves ~80 for the root on a 260-character limit. Checked on the names, before compressing. |
| 9 | `7z x` into `<runner.temp>\Client Test\Downloads\Pose3D-Windows\` | **the step the old gate did not have.** A space, because that is where an unquoted path in a script splits in two; a nesting level, because the client's root has one; `7z`, because that is what wrote the archive and what they open it with — `Expand-Archive` is .NET's `ZipArchive`, minutes on 1.6 GB and with long-path behaviour of its own. |
| 10 | `check_bundle_layout.py` **on the extracted copy** | a file the archive or the extraction dropped. This is the copy that actually broke last time. |
| 11 | `Pose3D.exe --selftest --no-video`, from the extracted copy, `Start-Process -Wait` | the app the client double-clicks, from where they run it, with `POSE3D_BLENDER=""`, `POSE3D_MODELS=""` and an empty `XDG_CACHE_HOME`: it can only pass by finding what the bundle itself ships. `Start-Process -Wait` and not `&`, because `Pose3D.exe` is built for the GUI subsystem and PowerShell does not wait for those — `&` returns immediately and `$LASTEXITCODE` would describe the launch, not the self-test. |
| 12 | print `selftest-out.txt`, `selftest-err.txt` and the extracted copy's `pose3d-log.txt`, then fail on a non-zero exit | a windowed exe writes into `pose3d-log.txt` and nowhere else. Printed on a *passing* run too, so the log records which Blender and which weights the bundle actually found. |
| 13 | `Pose3D-diagnose.exe --selftest --no-video`, same starved environment, wrapped in `try`/`catch`, whenever the extraction succeeded | the console build of the same checks — the exe the client is told to run when something goes wrong. Running it here, pass or fail, means the report they would send is one this gate has already produced. Evidence, not a second gate: step 12 already failed the job on that exit code, and this step is meant to run *after* that failure — hence the `try`/`catch` and the `exit 0`: under GitHub's `$ErrorActionPreference = 'Stop'` a non-zero exit from a native command is itself terminating, which would turn a green job's evidence step red. `--no-video` for the same reason step 12 has it: EEVEE needs WGL extensions this runner has no driver for, and without the flag this asks a slower, different question. |

## The one check this runner cannot make

The GitHub runner has no GPU. Qt cannot create a hardware context, and the
software renderer the bundle ships (`opengl32sw.dll`) drives **Qt** but not
pyqtgraph, which draws through PyOpenGL and loads the machine's own
`opengl32.dll`. So step 11 keeps `POSE3D_NO_GL=1`, and `check_qt_opengl`
reports *Degraded* instead of failing.

That is a property of the machine, not of the build, and it is the only check
in the gate that is weakened. What stands in for it:

* the manifest **requires** `_internal\PySide6\opengl32sw.dll` and
  `_internal\PySide6\plugins\platforms\qwindows.dll` — in the **extracted**
  copy (step 10), so a 3D view missing from the *package* still fails here;
* `check_qt_opengl`'s `ImportError` path stays fatal, so a build that cannot
  even import the 3D view fails whatever the GPU situation;
* on the client's machine `Pose3D.exe --selftest` is strict, and
  `Diagnose.cmd` (which runs `Pose3D-diagnose.exe --diagnose`) is how they run
  it — with no arguments that exe diagnoses rather than starting the app.

## What the gate still does not prove

* **No signature.** The build is unsigned, so SmartScreen shows *"Windows
  protected your PC"* on first run. Documented in `README.txt`; see
  `docs/DECISIONS.md`.
* **No second Windows machine.** The gate runs on `windows-latest` only. A
  fault specific to an older Windows, a different locale or a machine with an
  aggressive antivirus is still discovered by the client.
* **No preview video.** EEVEE needs WGL extensions this runner has no driver
  for, so `--no-video`; `check_video` is Degraded here anyway.
* **Nothing about accuracy.** That is `windows-test.yml`, which runs the whole
  test suite — including the Blender export path — with `POSE3D_REQUIRE_*` set
  so the export tests cannot skip themselves into a green run.
