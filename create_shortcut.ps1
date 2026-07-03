param(
    [string]$ShortcutName = "Raman Tool 快速启动",
    [string]$ShortcutDirectory = [Environment]::GetFolderPath("Desktop")
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PyvenvCfg = Join-Path $ProjectRoot ".venv\pyvenv.cfg"
$Launcher = Join-Path $ProjectRoot "launch_raman_tool.pyw"
$FallbackPythonw = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
$IconPath = Join-Path $ProjectRoot ".venv\Scripts\raman-tool-qt.exe"

if (-not (Test-Path -LiteralPath $PyvenvCfg)) {
    throw "未找到 .venv\pyvenv.cfg。请先在项目根目录运行: uv sync --extra qt"
}

if (-not (Test-Path -LiteralPath $Launcher)) {
    throw "未找到 launch_raman_tool.pyw。请确认仓库文件完整。"
}

$HomeLine = Get-Content -LiteralPath $PyvenvCfg | Where-Object { $_ -match '^home\s*=\s*(.+)$' } | Select-Object -First 1
$PythonHome = if ($HomeLine -match '^home\s*=\s*(.+)$') { $Matches[1].Trim() } else { $null }
$RealPythonw = if ($PythonHome) { Join-Path $PythonHome "pythonw.exe" } else { $null }

if ($RealPythonw -and (Test-Path -LiteralPath $RealPythonw)) {
    $Pythonw = $RealPythonw
} elseif (Test-Path -LiteralPath $FallbackPythonw) {
    $Pythonw = $FallbackPythonw
} else {
    throw "未找到 pythonw.exe。请先安装 Python 3.10+ 并运行: uv sync --extra qt"
}

if (-not (Test-Path -LiteralPath $ShortcutDirectory)) {
    New-Item -ItemType Directory -Path $ShortcutDirectory | Out-Null
}

$ShortcutPath = Join-Path $ShortcutDirectory ($ShortcutName + ".lnk")
$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $Pythonw
$Shortcut.Arguments = '"' + $Launcher + '"'
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.WindowStyle = 1
if (Test-Path -LiteralPath $IconPath) {
    $Shortcut.IconLocation = $IconPath + ",0"
}
$Shortcut.Save()

Write-Host "已创建快捷方式: $ShortcutPath"
Write-Host "启动器: $Pythonw"