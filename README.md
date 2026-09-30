# Raman Tool

Raman Tool is a local Raman spectroscopy data processing application. The project now keeps a single desktop GUI: the Qt interface built with PySide6. Command-line and terminal workflows remain available for scripting and batch processing.

## Features

- Qt desktop GUI for loading, viewing, processing, and exporting spectra.
- CLI commands for plotting, batch processing, SNR calculation, baseline correction, concentration analysis, and file inspection.
- Terminal UI for lightweight interactive workflows.
- Supported input formats: `.txt`, `.asc`, `.sif`, `.tif`, `.tiff`, `.bmp`, `.jpg`, `.jpeg`.
- Image import options for row groups, column merging, calibration, and mean/sum row mode.
- Workflow presets for reusable baseline, concentration, batch, and display settings.
- Repeatable baseline correction with an immutable raw-data snapshot, undo/redo, restore, and an append-only processing history.
- Configurable gas Raman peak library for reference markers, automatic peak matching, and concentration windows.
- Safety limits for very large files, oversized images, excessive data points, and expensive baseline correction.
- Export protection: existing outputs are not overwritten by default; GUI exports ask for confirmation.

## Requirements

- Python 3.10 or newer
- Windows, Linux, or macOS
- `uv` is recommended for dependency management

## Installation

Install `uv` from the official Astral documentation:

<https://docs.astral.sh/uv/getting-started/installation/>

For better supply-chain safety, review installer scripts before running them and prefer pinned/locked installs when distributing the tool.

Clone or copy the project, then install dependencies:

```powershell
cd Raman_tool
uv sync --extra qt
```

For development and tests:

```powershell
uv sync --extra qt --extra dev
```

## Start The Qt GUI

```powershell
uv run raman-tool-qt
```

Equivalent module command:

```powershell
uv run python -m raman_tool qt
```

On Windows, the repository also includes a no-console launcher:

```powershell
uv sync --extra qt
powershell -ExecutionPolicy Bypass -File .\create_shortcut.ps1
```

This creates a desktop shortcut that starts:

```text
pythonw.exe -> launch_raman_tool.pyw -> raman_tool.qt_gui
```

## Command Line

Show help:

```powershell
uv run raman-tool --help
```

Common examples:

```powershell
uv run raman-tool plot test_data\demo.txt
uv run raman-tool plot test_data\demo.txt --overwrite
uv run raman-tool batch test_data --baseline -o output
uv run raman-tool snr test_data\demo.txt --peak 2300,2350 --noise 3500,4000
uv run raman-tool baseline test_data\demo.txt -m poly -d 3
uv run raman-tool concentration test_data\demo.txt O2
uv run raman-tool info test_data\demo.txt
uv run raman-tool tui
uv run raman-tool qt
```

Output files are protected from accidental overwrite by default. Use `--overwrite` on commands that support it when replacing an existing output is intended.

## Supported Formats

| Format | Description |
| --- | --- |
| `.txt` | Text spectrum data, either one intensity column or at least two columns `(x, intensity)` |
| `.asc` | ASC text data; non-numeric header lines are skipped |
| `.sif` | Andor Solis SIF spectra |
| `.tif` / `.tiff` | TIFF image spectra |
| `.bmp` | BMP image spectra |
| `.jpg` / `.jpeg` | JPEG image spectra |

## Coordinate Units

The application distinguishes an uncalibrated pixel axis (`px`) from a Raman-shift axis (`cm⁻¹`) throughout loading, plotting, peak tables, status messages, and data export.

- Exported TXT/ASC headers preserve units, calibration and history on reimport; legacy two-column files without a unit header are treated as Raman shift data.
- SIF uses Raman shift only when a complete explicit calibration is available. Missing calibration stays in pixels; no coefficients or laser wavelength are guessed.
- One-column TXT/ASC data and image spectra use pixels until a linear calibration is supplied.
- Gas reference peaks and quantitative analysis are defined in `cm⁻¹`. They are disabled for pixel-axis spectra to prevent a pixel position from being mistaken for a Raman shift.
- Exported TXT/ASC headers include the actual x-axis unit and record the linear calibration when one is present.

## Configuration

Runtime settings are stored in a TOML file. By default the app uses `%APPDATA%\\RamanTool\\config.toml` on Windows and `~/.raman_tool/config.toml` elsewhere. Set `RAMAN_TOOL_CONFIG` to point to a custom config file.

In the Qt GUI, open `设置 -> 安全限制...` to edit the main safety limits.

Workflow presets are stored next to the config file as `presets.json` by default. Set `RAMAN_TOOL_PRESETS` to use a custom preset file. In the Qt GUI, use `预设 -> 应用工作流预设...` or `预设 -> 保存当前为预设...`.

The gas peak library is stored next to the config file as `gas_library.json` by default. Set `RAMAN_TOOL_GAS_LIBRARY` to use a custom library file. In the Qt GUI, open `设置 -> 气体峰位库...`.

## Workflow Presets

Workflow presets capture frequently changed processing settings:

- Baseline method, arPLS lambda, polynomial degree, and automatic baseline toggles
- Concentration focus gas, peak window, and peak height/area strategy
- Batch baseline toggle
- Image row mode, column merge factor, gas peak markers, and automatic peak markers

Built-in presets include `标准气体分析`, `峰面积定量`, and `快速查看`. User presets saved from the Qt GUI are written to `presets.json`.

## Traceable Baseline Processing

Each loaded spectrum keeps its original Raman shift and intensity arrays as a protected session snapshot. Manual baseline correction always operates on the current state, so correction can be applied repeatedly when required. Use `Ctrl+Z`, `Ctrl+Y`, or the controls in the baseline panel to undo, redo, or restore the original spectrum.

Automatic loading correction, manual corrections, undo, redo, and restore operations are recorded in the processing history. Exporting the current spectrum embeds this history in a comment header. Use `File -> Export Original Spectrum Data...` to export the untouched session snapshot separately.

## Gas Peak Library

The configurable gas peak library controls reference peak markers, automatic peak-to-gas matching, the default concentration windows, and single-file and batch quantitative outputs. Each gas entry contains a case-preserving key such as `CBrF₃`, display name, Raman peak center in `cm⁻¹`, half-window width in `cm⁻¹`, correction coefficient, color, enabled flag, and quantitative flag.

`O2`, `N2`, and `CO2` are quantitative in the default library, but they are not hard-coded into the processing chain. Every enabled entry marked as quantitative appears dynamically in the Qt selector, single-file results, batch concentration curves, and TXT exports. Disabled or non-quantitative entries are excluded. If no quantitative gas is enabled, the application reports the configuration problem instead of silently falling back to a built-in list.

## Safety Limits

Default safety limits are defined in `src/raman_tool/config.py` and loaded through `src/raman_tool/safety.py`.

- Maximum generic input file size: `512 MB`
- Maximum text input file size: `128 MB`
- Maximum spectrum data points: `2,000,000`
- Minimum supported image size target: `2048 x 2048`
- Current image pixel limit: `4096 x 4096`
- Image array value limit: pixel limit times 4 channels
- Maximum arPLS baseline points: `200,000`

These limits are intended to prevent accidental memory or CPU exhaustion when opening malformed or unexpectedly large files.

## Project Layout

```text
Raman_tool/
├── pyproject.toml
├── README.md
├── launch_raman_tool.pyw
├── create_shortcut.ps1
├── src/
│   └── raman_tool/
│       ├── cli.py
│       ├── qt_gui.py
│       ├── tui.py
│       ├── safety.py
│       ├── readers/
│       ├── processing/
│       └── visualization/
├── tests/
└── test_data/
```

## Tests

```powershell
uv run pytest
```

The current test suite covers core models, readers, processing, sorting, exporters, and safety limits.

## Consistency and Reference Validation

The desktop, CLI and terminal use common loading and validation functions.
Finite one-dimensional data and strictly monotonic coordinates are required;
ascending and descending axes are both supported. Column merging retains
original detector-bin centers, including the final partial bin. Text export uses
17 significant digits to preserve float64 values on reimport.

Both peak height and area subtract the line joining window endpoints. Quantitative
results are normalized weighted signals within the enabled gas set; default
coefficients are all 1. They require independent response calibration before
interpretation as physical composition. Missing windows, insufficient points,
invalid settings and zero total signal produce explicit errors.

Manual/automatic baseline operations run in background tasks. Loads ignore stale
results; batch tasks capture their gas library and selected peak-window override.
The same settings produce the same single/batch result. Cancellation takes effect
between reads and numerical operations.

History defaults to 32 snapshots / 128 MiB, preserving original data and a full
audit stream. The interface displays the latest 200 audit records. The session
cache retains 16 resident sessions / 256 MiB and spills others to private temporary
disk storage (1 GiB). Configure budgets under [runtime] in config.toml.
These resources last for the running application; export results before exit.

~~~powershell
uv run raman-tool concentration sample.asc CO2 --strategy peak_area --baseline --lam 100000
uv run raman-tool info image.tif --col-merge 2 --calibration 2,100
uv run raman-tool validate-standard tests/fixtures/standards/synthetic-mixture.json --report output/synthetic-report.json
~~~

The reference validator records source hashes, parameters, declared truth and
deviations. The included 78:21:1 fixture is synthetic. Regression tests cover
all four calculation entry points and queued Qt workers. CI configuration adds
Windows/Linux, Python 3.10/3.12, minimum dependencies and package builds.
Remote CI executes after pushing changes; local checks do not imply remote success.

See [validation and release instructions](docs/VALIDATION.md) for measured samples,
report interpretation, limits and release checks.

## GUI Policy

Only the Qt desktop GUI is maintained. Start it with `raman-tool-qt` or `raman-tool qt`.

## License

MIT License
