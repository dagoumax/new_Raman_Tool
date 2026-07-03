# Raman Tool 拉曼光谱数据处理工具

Raman Tool 是一个用于拉曼光谱数据读取、可视化、信噪比计算、基线校正、气体浓度分析和批量处理的桌面/命令行工具。

## 功能

- 多格式读取：TXT、ASC、SIF、TIFF、BMP
- 数据可视化：绘制光谱图，支持 PNG/PDF/SVG 导出
- 基线校正：支持多项式拟合和 airPLS 算法
- 信噪比计算：可指定信号区域和噪声区域
- 气体浓度分析：支持 N2、O2、CO2、H2、CH4 等气体
- 批量处理：批量读取并处理目录中的光谱文件
- 图形界面：Qt 桌面界面，支持拖拽导入
- 命令行：适合脚本化和批处理

## 系统要求

- Python >= 3.10
- Windows / Linux / macOS
- 推荐使用 uv 管理依赖

## 安装

### 1. 安装 uv

Windows PowerShell:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 克隆项目

```bash
git clone <your-repo-url>
cd Raman_tool
```

### 3. 创建虚拟环境并安装依赖

推荐安装 Qt 图形界面依赖：

```bash
uv sync --extra qt
```

如果只需要基础命令行功能：

```bash
uv sync
```

## 启动方式

### Qt 图形界面

```bash
uv run raman-tool-qt
```

或：

```bash
uv run python -m raman_tool qt
```

### 命令行

```bash
uv run raman-tool --help
```

常用示例：

```bash
uv run raman-tool plot data/spectrum.txt
uv run raman-tool batch data/ --baseline -o output/
uv run raman-tool snr data/spectrum.txt --peak 2300,2350 --noise 3500,4000
uv run raman-tool baseline data/spectrum.txt -m poly -d 3
uv run raman-tool concentration data/spectrum.txt H2 --ref N2 --ref-conc 78.0
uv run raman-tool info data/spectrum.txt
```

## Windows 创建桌面快捷启动方式

仓库中提供了两个文件用于无终端启动：

- `launch_raman_tool.pyw`：真正启动 Qt GUI 的无控制台脚本
- `create_shortcut.ps1`：自动在桌面创建快捷方式

### 推荐方式：运行脚本自动创建

在项目根目录运行：

```powershell
uv sync --extra qt
powershell -ExecutionPolicy Bypass -File .\create_shortcut.ps1
```

运行成功后，桌面会出现：

```text
Raman Tool 快速启动.lnk
```

双击该快捷方式即可启动图形界面。

### 为什么不用 .venv\Scripts\pythonw.exe

如果虚拟环境由 uv 创建，`.venv\Scripts\pythonw.exe` 可能是一个轻量转发启动器。部分 Windows 环境会把它当成控制台程序处理，导致启动 GUI 时额外弹出黑色终端窗口。

`create_shortcut.ps1` 会读取 `.venv\pyvenv.cfg` 中的 `home = ...`，优先找到 uv 安装的真实 Python GUI 解释器：

```text
<python-home>\pythonw.exe
```

快捷方式最终使用：

```text
真实 pythonw.exe -> launch_raman_tool.pyw -> raman_tool.qt_gui
```

这样可以避免启动时出现终端窗口。

### 手动创建快捷方式

如果需要手动创建，右键桌面新建快捷方式：

目标填写真实的 `pythonw.exe`，例如：

```text
C:\Users\你的用户名\AppData\Roaming\uv\python\cpython-3.10-windows-x86_64-none\pythonw.exe
```

参数填写：

```text
"C:\path\to\Raman_tool\launch_raman_tool.pyw"
```

起始位置填写项目根目录：

```text
C:\path\to\Raman_tool
```

真实 `pythonw.exe` 的路径可以从 `.venv\pyvenv.cfg` 的 `home = ...` 找到。

## 支持的文件格式

| 格式 | 描述 |
| --- | --- |
| `.txt` | 两列文本数据，通常为拉曼位移和强度 |
| `.asc` | ASC 文本数据 |
| `.sif` | Andor Solis 光谱文件 |
| `.tif` / `.tiff` | TIFF 图像 |
| `.bmp` | BMP 图像 |

## 项目结构

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
│       ├── gui.py
│       ├── tui.py
│       ├── readers/
│       ├── processing/
│       └── visualization/
├── tests/
└── test_data/
```

## 测试

```bash
uv run pytest
```

## 迁移到其他电脑

1. 安装 Python 3.10+ 和 uv
2. 克隆仓库
3. 在项目根目录运行 `uv sync --extra qt`
4. Windows 用户运行 `powershell -ExecutionPolicy Bypass -File .\create_shortcut.ps1`
5. 使用桌面快捷方式启动，或运行 `uv run raman-tool-qt`

## License

MIT License
