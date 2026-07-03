"""图表绘制模块.

使用 matplotlib 生成光谱图表。
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import sys
from raman_tool.models import Spectrum

# 气体拉曼峰参考数据 (cm⁻¹)
GAS_PEAKS = {
    2331.0: "N2",
    1555.0: "O2",
    1388.0: "CO2",
    3657.0: "H2O",
    2917.0: "CH4",
    4155.0: "H2",
    2143.0: "CO",
    1151.0: "SO2",
    1876.0: "NO",
    3334.0: "NH3",
    2954.0: "C2H6",
    1285.0: "CO2(v2)",
}

# 颜色映射
GAS_COLORS = {
    "N2": "#2c3e50", "O2": "#e74c3c", "CO2": "#27ae60",
    "H2O": "#3498db", "CH4": "#8e44ad", "H2": "#f39c12",
    "CO": "#1abc9c", "SO2": "#c0392b", "NO": "#7f8c8d",
    "NH3": "#2ecc71", "C2H6": "#d35400",
}


def _add_gas_peaks(ax, x_min: float, x_max: float, detected_positions: set | None = None):
    """在坐标轴上叠加参考气体峰位 (淡虚线，仅显示未被自动匹配的气体)."""
    y_min, y_max = ax.get_ylim()
    for pos, name in sorted(GAS_PEAKS.items()):
        if not (x_min <= pos <= x_max):
            continue
        # 如果该位置已有实测峰匹配，跳过参考线（避免重复标注）
        if detected_positions and pos in detected_positions:
            continue
        color = GAS_COLORS.get(name, "#999")
        ax.axvline(pos, color=color, linestyle=":", linewidth=0.6, alpha=0.4)
        ax.text(pos, y_max * 0.92, name, color=color, fontsize=6,
                ha="center", va="top", rotation=90, alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.05", fc="white", ec=color, alpha=0.6))


def _add_detected_peaks(ax, peaks: list[dict], x_min: float, x_max: float):
    """叠加自动寻峰结果 (顶点圆点 + 气体名/位置标注)."""
    y_min, y_max = ax.get_ylim()
    for p in peaks:
        cx = p["center"]
        ch = p["height"]
        if not (x_min <= cx <= x_max):
            continue

        gas = p.get("matched_gas")
        if gas:
            # 已匹配气体: 用该气体颜色 + 名称
            from raman_tool.visualization.plot import GAS_COLORS
            color = GAS_COLORS.get(gas, "#e67e22")
            # 圆点标记峰顶
            ax.plot(cx, ch, "o", color=color, markersize=6, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.8)
            # 标注: 气体名 + 位置
            label = f"{gas}\n{cx:.1f}"
            ax.text(cx, ch + (y_max - y_min) * 0.03, label,
                    color=color, fontsize=7, ha="center", va="bottom",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.9))
        else:
            # 未匹配: 灰色小圆点 + 位置数值
            ax.plot(cx, ch, "o", color="#888", markersize=4, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.text(cx, ch + (y_max - y_min) * 0.02, f"{cx:.1f}",
                    color="#888", fontsize=6, ha="center", va="bottom",
                    rotation=90)

# 配置中文字体支持
if sys.platform == "win32":
    for font_name in ["Microsoft YaHei", "SimHei", "SimSun"]:
        for f in fm.fontManager.ttflist:
            if f.name == font_name:
                matplotlib.rcParams["font.sans-serif"] = [font_name]
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
        if matplotlib.rcParams["font.sans-serif"] != ["DejaVu Sans"]:
            break


def plot_spectrum(
    spectrum: Spectrum,
    title: str | None = None,
    xlabel: str = "拉曼位移 (cm-1)",
    ylabel: str = "强度 (a.u.)",
    color: str = "steelblue",
    linewidth: float = 0.8,
    figsize: tuple = (10, 6),
    show: bool = False,
    show_individual_rows: bool = True,
    show_gas_peaks: bool = False,
    detected_peaks: list[dict] | None = None,
) -> plt.Figure:
    """绘制光谱。

    如果 spectrum.metadata 中有 "individual_rows"，则叠加绘制各行数据
    (半透明细线) + 均值 (粗线)。

    Args:
        spectrum: 光谱数据
        title: 图表标题
        xlabel: x 轴标签
        ylabel: y 轴标签
        color: 均值线条颜色
        linewidth: 均值线宽
        figsize: 图像尺寸
        show: 是否显示
        show_individual_rows: 是否叠加各行数据

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    rows = spectrum.metadata.get("individual_rows")
    if rows and show_individual_rows:
        n_rows = len(rows)
        total_rows = spectrum.metadata.get("total_rows", n_rows)
        for i, row_data in enumerate(rows):
            alpha = max(0.08, 0.35 * (1.0 - i / (n_rows + 10)))
            ax.plot(
                spectrum.raman_shift, row_data,
                color="gray", linewidth=0.35, alpha=alpha,
            )
        # 均值粗线
        label = f"均值 (n={n_rows})"
        if total_rows > n_rows:
            label = f"均值 (显示 {n_rows}/{total_rows} 行)"
        ax.plot(spectrum.raman_shift, spectrum.intensity,
                color=color, linewidth=linewidth + 0.6, label=label)
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.plot(spectrum.raman_shift, spectrum.intensity,
                color=color, linewidth=linewidth)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(spectrum.filename or "拉曼光谱")
    ax.grid(True, alpha=0.3)

    x_rng_min = float(np.min(spectrum.raman_shift))
    x_rng_max = float(np.max(spectrum.raman_shift))

    # 参考气体峰位标注 (淡虚线, 跳过已匹配的)
    if show_gas_peaks:
        detected_positions = None
        if detected_peaks:
            detected_positions = {p["center"] for p in detected_peaks if p.get("matched_gas")}
        _add_gas_peaks(ax, x_rng_min, x_rng_max, detected_positions)

    # 自动寻峰标注 (彩色圆点 + 气体名/位置)
    if detected_peaks:
        _add_detected_peaks(ax, detected_peaks, x_rng_min, x_rng_max)

    fig.tight_layout()

    if show:
        plt.show()

    return fig


def plot_baseline(
    spectrum: Spectrum,
    baseline: np.ndarray,
    corrected: Spectrum | None = None,
    title: str | None = None,
    figsize: tuple = (10, 8),
    show: bool = False,
) -> plt.Figure:
    """绘制光谱及其基线校正结果.

    Args:
        spectrum: 原始光谱
        baseline: 基线数组
        corrected: 校正后的光谱 (可选)
        title: 图表标题
        figsize: 图像尺寸
        show: 是否显示图像

    Returns:
        matplotlib Figure 对象
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax1 = axes[0]
    ax1.plot(spectrum.raman_shift, spectrum.intensity, "b-", linewidth=0.8, label="原始光谱")
    ax1.plot(spectrum.raman_shift, baseline, "r--", linewidth=1.2, label="拟合基线")
    ax1.set_ylabel("强度 (a.u.)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if title:
        ax1.set_title(f"{title} - 基线拟合")

    if corrected is not None:
        ax2 = axes[1]
        ax2.plot(corrected.raman_shift, corrected.intensity, "g-", linewidth=0.8)
        ax2.set_xlabel("拉曼位移 (cm-1)")
        ax2.set_ylabel("强度 (a.u.)")
        ax2.set_title("基线校正后")
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if show:
        plt.show()

    return fig


def plot_multiple(
    spectra: list[Spectrum],
    labels: list[str] | None = None,
    title: str | None = None,
    xlabel: str = "拉曼位移 (cm-1)",
    ylabel: str = "强度 (a.u.)",
    colors: list[str] | None = None,
    figsize: tuple = (12, 6),
    show: bool = False,
    offset: float = 0.0,
) -> plt.Figure:
    """绘制多条光谱进行比较.

    Args:
        spectra: 光谱列表
        labels: 每条光谱的标签
        title: 图表标题
        xlabel: x 轴标签
        ylabel: y 轴标签
        colors: 颜色列表
        figsize: 图像尺寸
        show: 是否显示图像
        offset: Y 轴偏移量 (用于垂直堆叠)

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = ["steelblue", "coral", "seagreen", "darkorange", "purple"]

    for i, spec in enumerate(spectra):
        color = colors[i % len(colors)]
        label = labels[i] if labels else spec.filename or f"光谱 {i + 1}"
        y_data = spec.intensity + i * offset
        ax.plot(spec.raman_shift, y_data, color=color, linewidth=0.8, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def save_figure(fig: plt.Figure, filepath: str | Path, dpi: int = 150) -> str:
    """保存图表到文件.

    Args:
        fig: matplotlib Figure 对象
        filepath: 输出文件路径 (支持 .png, .pdf, .svg)
        dpi: 分辨率

    Returns:
        实际保存的文件路径
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(filepath), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(filepath.absolute())
