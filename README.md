# Raman Tool

Raman Tool is a local Raman spectroscopy data processing application. The project now keeps a single desktop GUI: the Qt interface built with PySide6. Command-line and terminal workflows remain available for scripting and batch processing.

## Features

- Qt desktop GUI for loading, viewing, processing, and exporting spectra.
- CLI commands for plotting, batch processing, SNR calculation, baseline correction, concentration analysis, and file inspection.
- Terminal UI for lightweight interactive workflows.
- Supported input formats: `.txt`, `.asc`, `.sif`, `.tif`, `.tiff`, `.bmp`, `.jpg`, `.jpeg`.
- Image import options for row groups, column merging, calibration, and mean/sum row mode.
- Workflow presets for reusable baseline, concentration, batch, and display settings.
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
uv run raman-tool concentration test_data\demo.txt H2 --ref N2 --ref-conc 78.0
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

## Gas Peak Library

The configurable gas peak library controls reference peak markers, automatic peak-to-gas matching, the default concentration windows, and batch quantitative outputs. Each gas entry contains a case-preserving key such as `CBrF₃`, display name, Raman peak center, half-window width, correction coefficient, color, enabled flag, and quantitative flag.

`O2`, `N2`, and `CO2` are quantitative by default. Any additional gas marked as quantitative is included in batch concentration curves and TXT exports.

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

## GUI Policy

Only the Qt desktop GUI is maintained. Start it with `raman-tool-qt` or `raman-tool qt`.

## License

MIT License
