"""命令行接口(CLI).

提供批量处理、单文件分析、信噪比计算、浓度计算等功能。
用法:
    raman-tool plot <file> [--row-groups X] [--col-merge N]
    raman-tool batch <dir> [--baseline]
    raman-tool snr <file> [--peak X,Y] [--noise X,Y]
    raman-tool baseline <file> [--method arPLS] [--lam 1e5]
    raman-tool concentration <file> <gas> [--ref N2]
    raman-tool info <file> [--row-groups X] [--col-merge N]
    raman-tool tui / qt
"""

import argparse
import sys
from pathlib import Path
from raman_tool.models import Spectrum
from raman_tool.readers import read_file, SUPPORTED_FORMATS
from raman_tool.sorting import natural_sorted
from raman_tool.processing import (
    calculate_snr,
    calculate_concentration,
    subtract_baseline,
)
from raman_tool.visualization import plot_spectrum, plot_baseline, plot_multiple, save_figure
from raman_tool.exporters import unique_path


def _resolve_output_path(path: Path | str, overwrite: bool = False) -> Path:
    path = Path(path)
    if overwrite or not path.exists():
        return path
    return unique_path(path)


def _load_spectrum(args: argparse.Namespace) -> Spectrum:
    return read_file(
        Path(args.file),
        row_groups=getattr(args, "row_groups", None),
        col_merge=getattr(args, "col_merge", 1),
    )


import matplotlib
matplotlib.use("Agg")

def cmd_plot(args: argparse.Namespace) -> int:
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"错误: 文件不存在 {filepath}", file=sys.stderr)
        return 1

    spectrum = _load_spectrum(args)
    if args.range:
        parts = args.range.split(",")
        if len(parts) == 2:
            spectrum = spectrum.crop(float(parts[0]), float(parts[1]))

    fig = plot_spectrum(spectrum, show=not args.no_show)
    if args.output:
        path = save_figure(fig, _resolve_output_path(args.output, args.overwrite))
        print(f"图表已保存 {path}")
    else:
        out = filepath.with_suffix(".png")
        path = save_figure(fig, _resolve_output_path(out, args.overwrite))
        print(f"图表已保存 {path}")

    return 0

import matplotlib
matplotlib.use("Agg")

def cmd_batch(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"错误: 目录不存在 {directory}", file=sys.stderr)
        return 1

    supported_exts = set(SUPPORTED_FORMATS.keys())
    all_files = []
    for ext in supported_exts:
        all_files.extend(directory.glob(f"*{ext}"))
        all_files.extend(directory.glob(f"*{ext.upper()}"))
    all_files = natural_sorted(all_files)

    if not all_files:
        print(f"错误: 在{directory} 中未找到支持格式的文件", file=sys.stderr)
        return 1

    out_dir = Path(args.output) if args.output else directory / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"找到{len(all_files)}个文件，输出目录: {out_dir}")

    ok, fail = 0, 0
    for filepath in all_files:
        try:
            spectrum = read_file(filepath)
            if args.baseline:
                spectrum = subtract_baseline(spectrum, method="arPLS")
            fig = plot_spectrum(spectrum, show=False)
            out_path = _resolve_output_path(out_dir / f"{filepath.stem}.png", args.overwrite)
            save_figure(fig, out_path)
            ok += 1
            print(f"  OK {filepath.name}")
        except Exception as e:
            fail += 1
            print(f"  FAIL {filepath.name}: {e}")

    print(f"完成: {ok} 成功, {fail} 失败")
    return 0 if fail == 0 else 1


def cmd_snr(args: argparse.Namespace) -> int:
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"错误: 文件不存在 {filepath}", file=sys.stderr)
        return 1

    spectrum = _load_spectrum(args)

    peak_start = peak_end = noise_start = noise_end = None
    if args.peak:
        parts = args.peak.split(",")
        if len(parts) == 2:
            peak_start, peak_end = float(parts[0]), float(parts[1])
    if args.noise:
        parts = args.noise.split(",")
        if len(parts) == 2:
            noise_start, noise_end = float(parts[0]), float(parts[1])

    result = calculate_snr(spectrum, peak_start, peak_end, noise_start, noise_end)

    print(f"文件: {filepath.name}")
    print(f"信噪比(SNR): {result['snr']:.2f}")
    print(f"信号强度:    {result['signal']:.2f}")
    print(f"噪声 RMS:     {result['noise_rms']:.4f}")
    print(f"峰中心位置:  {result['peak_center']:.2f} cm-1")
    if result["peak_area"] > 0:
        print(f"峰面积:      {result['peak_area']:.4f}")
    return 0


import matplotlib
matplotlib.use("Agg")

def cmd_baseline(args: argparse.Namespace) -> int:
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"错误: 文件不存在 {filepath}", file=sys.stderr)
        return 1

    spectrum = _load_spectrum(args)

    method = args.method or "arPLS"
    lam = args.lam or 1e5

    corrected = subtract_baseline(spectrum, method=method, lam=lam, degree=3)
    baseline = spectrum.intensity - corrected.intensity

    fig = plot_baseline(spectrum, baseline=baseline, corrected=corrected, show=not args.no_show)
    out = Path(args.output) if args.output else filepath.with_stem(f"{filepath.stem}_baseline").with_suffix(".png")
    path = save_figure(fig, _resolve_output_path(out, args.overwrite))
    print(f"图表已保存 {path}")
    return 0


def cmd_concentration(args: argparse.Namespace) -> int:
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"错误: 文件不存在 {filepath}", file=sys.stderr)
        return 1

    spectrum = _load_spectrum(args)

    try:
        result = calculate_concentration(
            spectrum,
            gas_name=args.gas,
            window=args.window or 10.0,
            reference_gas=args.ref or "N2",
            reference_concentration=args.ref_conc or 78.0,
        )
        print(f"文件: {filepath.name}")
        print(f"目标气体: {result['gas']}")
        print(f"浓度:     {result['concentration']:.4f} %")
        print(f"峰中心:   {result['peak_center']:.2f} cm-1")
        print(f"峰面积:   {result['peak_area']:.4f}")
        print(f"参考气体: {result['reference_gas']}")
        print(f"参考峰:   {result['reference_peak_center']:.2f} cm-1")
        print(f"参考面积: {result['reference_peak_area']:.4f}")
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"错误: 文件不存在 {filepath}", file=sys.stderr)
        return 1

    spectrum = _load_spectrum(args)

    print(f"文件:        {filepath.name}")
    print(f"格式:        {filepath.suffix.upper()}")
    print(f"数据点数:    {spectrum.size}")
    print(f"拉曼位移范围: {spectrum.raman_shift[0]:.2f} - {spectrum.raman_shift[-1]:.2f} cm-1")
    print(f"强度范围:    {spectrum.intensity.min():.2f} - {spectrum.intensity.max():.2f}")
    print(f"强度均值:    {spectrum.intensity.mean():.2f}")
    print(f"强度标准差:  {spectrum.intensity.std():.2f}")

    meta = spectrum.metadata
    for key in ["image_shape", "selected_rows", "output_cols", "row_groups", "col_merge", "format"]:
        if key in meta:
            print(f"{key}:         {meta[key]}")

    return 0


def _add_tif_args(parser):
    parser.add_argument("--row-groups", help="TIF/BMP 行分组 如\"1-40, 91-130\"")
    parser.add_argument("--col-merge", type=int, default=1, help="TIF/BMP 列合并因子 (默认 1=不变)")


def main(args_list: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raman-tool",
        description="拉曼光谱数据处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"支持格式: {', '.join(SUPPORTED_FORMATS.keys())}",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # plot
    p_plot = subparsers.add_parser("plot", help="绘制光谱")
    p_plot.add_argument("file", help="光谱文件路径")
    p_plot.add_argument("-o", "--output", help="输出图片路径")
    p_plot.add_argument("-r", "--range", help="拉曼位移范围 (start,end)")
    p_plot.add_argument("--no-show", action="store_true", help="不显示图表")
    p_plot.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    _add_tif_args(p_plot)

    # batch
    p_batch = subparsers.add_parser("batch", help="批量处理")
    p_batch.add_argument("directory", help="包含数据文件的目录")
    p_batch.add_argument("-o", "--output", help="输出目录 (默认: output/)")
    p_batch.add_argument("--baseline", action="store_true", help="执行基线校正 (arPLS)")
    p_batch.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")

    # snr
    p_snr = subparsers.add_parser("snr", help="计算信噪比")
    p_snr.add_argument("file", help="光谱文件路径")
    p_snr.add_argument("--peak", help="信号峰区域 (start,end)")
    p_snr.add_argument("--noise", help="噪声区域 (start,end)")
    _add_tif_args(p_snr)

    # baseline
    p_bl = subparsers.add_parser("baseline", help="基线校正")
    p_bl.add_argument("file", help="光谱文件路径")
    p_bl.add_argument("-m", "--method", choices=["arPLS", "poly"], default="arPLS", help="基线方法 (默认: arPLS)")
    p_bl.add_argument("-d", "--degree", type=int, default=3, help="多项式阶数 (默认: 3)")
    p_bl.add_argument("--lam", type=float, default=1e5, help="arPLS 平滑参数 (默认 1e5)")
    p_bl.add_argument("-o", "--output", help="输出图片路径")
    p_bl.add_argument("--no-show", action="store_true", help="不显示图表")
    p_bl.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    _add_tif_args(p_bl)

    # concentration
    p_conc = subparsers.add_parser("concentration", help="计算气体浓度")
    p_conc.add_argument("file", help="光谱文件路径")
    p_conc.add_argument("gas", help="目标气体名称 (如 H2, CO2, CH4)")
    p_conc.add_argument("-w", "--window", type=float, default=10.0, help="峰搜索窗口宽度")
    p_conc.add_argument("--ref", default="N2", help="参考气体 (默认: N2)")
    p_conc.add_argument("--ref-conc", type=float, default=78.0, help="参考气体浓度 (默认: 78%%)")
    _add_tif_args(p_conc)

    # info
    p_info = subparsers.add_parser("info", help="显示光谱文件信息")
    p_info.add_argument("file", help="光谱文件路径")
    _add_tif_args(p_info)

    # tui / qt
    subparsers.add_parser("tui", help="启动终端交互界面")
    subparsers.add_parser("qt", help="启动 Qt 桌面界面")

    args = parser.parse_args(args_list)

    commands = {
        "plot": cmd_plot,
        "batch": cmd_batch,
        "snr": cmd_snr,
        "baseline": cmd_baseline,
        "concentration": cmd_concentration,
        "info": cmd_info,
    }

    if args.command == "tui":
        from raman_tool.tui import main as tui_main
        return tui_main()

    if args.command == "qt":
        from raman_tool.qt_gui import main as qt_main
        return qt_main()

    if args.command in commands:
        return commands[args.command](args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

