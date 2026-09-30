"""命令行接口(CLI).

提供批量处理、单文件分析、信噪比计算、浓度计算等功能。
用法:
    raman-tool plot <file> [--row-groups X] [--col-merge N]
    raman-tool batch <dir> [--baseline]
    raman-tool snr <file> [--peak X,Y] [--noise X,Y]
    raman-tool baseline <file> [--method arPLS] [--lam 1e5]
    raman-tool concentration <file> <gas>
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
from raman_tool.workflows import collect_spectrum_files, load_spectrum


def _resolve_output_path(path: Path | str, overwrite: bool = False) -> Path:
    path = Path(path)
    if overwrite or not path.exists():
        return path
    return unique_path(path)


def _load_spectrum(args: argparse.Namespace) -> Spectrum:
    return load_spectrum(
        Path(args.file),
        row_groups=getattr(args, "row_groups", None),
        col_merge=getattr(args, "col_merge", 1),
        calibration=getattr(args, "calibration", None),
        row_mode=getattr(args, "row_mode", "mean"),
    )


def cmd_plot(args: argparse.Namespace) -> int:
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"错误: 文件不存在 {filepath}", file=sys.stderr)
        return 1

    spectrum = _load_spectrum(args)
    if args.range:
        spectrum = spectrum.crop(*_parse_pair(args.range))

    fig = plot_spectrum(spectrum, show=not args.no_show)
    if args.output:
        path = save_figure(fig, _resolve_output_path(args.output, args.overwrite))
        print(f"图表已保存 {path}")
    else:
        out = filepath.with_suffix(".png")
        path = save_figure(fig, _resolve_output_path(out, args.overwrite))
        print(f"图表已保存 {path}")

    return 0

def cmd_batch(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"错误: 目录不存在 {directory}", file=sys.stderr)
        return 1

    all_files = collect_spectrum_files(directory)

    if not all_files:
        print(f"错误: 在{directory} 中未找到支持格式的文件", file=sys.stderr)
        return 1

    out_dir = Path(args.output) if args.output else directory / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"找到{len(all_files)}个文件，输出目录: {out_dir}")

    ok, fail = 0, 0
    for filepath in all_files:
        try:
            spectrum = load_spectrum(
                filepath, row_groups=args.row_groups, col_merge=args.col_merge,
                calibration=args.calibration, row_mode=args.row_mode,
                baseline_options=_baseline_options(args) if args.baseline else None,
            )
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
        peak_start, peak_end = _parse_pair(args.peak)
    if args.noise:
        noise_start, noise_end = _parse_pair(args.noise)

    result = calculate_snr(spectrum, peak_start, peak_end, noise_start, noise_end)

    print(f"文件: {filepath.name}")
    print(f"信噪比(SNR): {result['snr']:.2f}")
    print(f"信号强度:    {result['signal']:.2f}")
    print(f"噪声 RMS:     {result['noise_rms']:.4f}")
    print(f"峰中心位置:  {result['peak_center']:.2f} {spectrum.x_unit_label}")
    if result["peak_area"] > 0:
        print(f"峰面积:      {result['peak_area']:.4f}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"错误: 文件不存在 {filepath}", file=sys.stderr)
        return 1

    spectrum = _load_spectrum(args)

    corrected = subtract_baseline(spectrum, **_baseline_options(args))
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
        if args.baseline:
            spectrum = subtract_baseline(spectrum, **_baseline_options(args))
        result = calculate_concentration(
            spectrum,
            gas_name=args.gas,
            window=args.window,
            strategy=args.strategy,
        )
        print(f"文件: {filepath.name}")
        print(f"目标气体: {result['gas']}")
        print(f"归一化信号占比: {result['concentration']:.4f} %")
        print(f"峰中心:   {result['peak_center']:.2f} {spectrum.x_unit_label}")
        print(f"峰面积:   {result['peak_area']:.4f}")
        print("全部定量气体:")
        for gas, percentage in result["all_concentrations"]["percentages"].items():
            print(f"  {gas}: {percentage:.4f}%")
        for warning in result["all_concentrations"].get("warnings", []):
            print(f"说明: {warning}")
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
    print(
        f"横轴范围:    {spectrum.raman_shift[0]:.2f} - "
        f"{spectrum.raman_shift[-1]:.2f} {spectrum.x_unit_label}"
    )
    print(f"强度范围:    {spectrum.intensity.min():.2f} - {spectrum.intensity.max():.2f}")
    print(f"强度均值:    {spectrum.intensity.mean():.2f}")
    print(f"强度标准差:  {spectrum.intensity.std():.2f}")

    meta = spectrum.metadata
    for key in ["image_shape", "selected_rows", "output_cols", "row_groups", "col_merge", "format"]:
        if key in meta:
            print(f"{key}:         {meta[key]}")

    return 0


def _parse_pair(value: str) -> tuple[float, float]:
    try:
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError("参数必须是逗号分隔的两个数字，如 100,200") from exc


def cmd_validate_standard(args):
    from raman_tool.reference_validation import validate_standard, write_validation_report
    report = validate_standard(args.manifest)
    for gas, comparison in report["comparisons"].items():
        print(
            f"{gas}: 参考={comparison['expected_percent']:.6f}% "
            f"实算={comparison['actual_percent']:.6f}% "
            f"偏差={comparison['error_percentage_points']:.6f} 个百分点 "
            f"{'PASS' if comparison['passed'] else 'FAIL'}"
        )
    print(report["scope"])
    if args.report:
        write_validation_report(report, args.report, overwrite=args.overwrite)
        print(f"验证报告: {args.report}")
    return 0 if report["passed"] else 1


def _baseline_options(args):
    return {
        "method": args.method, "lam": args.lam,
        "degree": args.degree,
    }


def _add_baseline_args(parser):
    parser.add_argument("-m", "--method", choices=["arPLS", "poly"], default="arPLS")
    parser.add_argument("-d", "--degree", type=int, default=3)
    parser.add_argument("--lam", type=float, default=1e5)


def _add_tif_args(parser):
    parser.add_argument("--row-groups", help="TIF/BMP 行分组 如\"1-40, 91-130\"")
    parser.add_argument("--col-merge", type=int, default=1, help="TIF/BMP 列合并因子 (默认 1=不变)")
    parser.add_argument("--row-mode", choices=["mean", "sum"], default="mean")
    parser.add_argument("--calibration", type=_parse_pair, help="像素线性校准 a,b：x = a*pixel+b")


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
    p_plot.add_argument("-r", "--range", help="横轴范围 (start,end)")
    p_plot.add_argument("--no-show", action="store_true", help="不显示图表")
    p_plot.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    _add_tif_args(p_plot)

    # batch
    p_batch = subparsers.add_parser("batch", help="批量处理")
    p_batch.add_argument("directory", help="包含数据文件的目录")
    p_batch.add_argument("-o", "--output", help="输出目录 (默认: output/)")
    p_batch.add_argument("--baseline", action="store_true", help="执行基线校正 (arPLS)")
    p_batch.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    _add_tif_args(p_batch)
    _add_baseline_args(p_batch)

    # snr
    p_snr = subparsers.add_parser("snr", help="计算信噪比")
    p_snr.add_argument("file", help="光谱文件路径")
    p_snr.add_argument("--peak", help="信号峰区域 (start,end)")
    p_snr.add_argument("--noise", help="噪声区域 (start,end)")
    _add_tif_args(p_snr)

    # baseline
    p_bl = subparsers.add_parser("baseline", help="基线校正")
    p_bl.add_argument("file", help="光谱文件路径")
    _add_baseline_args(p_bl)
    p_bl.add_argument("-o", "--output", help="输出图片路径")
    p_bl.add_argument("--no-show", action="store_true", help="不显示图表")
    p_bl.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    _add_tif_args(p_bl)

    # concentration
    p_conc = subparsers.add_parser("concentration", help="计算气体浓度")
    p_conc.add_argument("file", help="光谱文件路径")
    p_conc.add_argument("gas", help="气体库中已启用定量的目标气体")
    p_conc.add_argument("-w", "--window", type=float, default=None, help="覆盖聚焦气体的半窗口 (cm⁻¹)")
    p_conc.add_argument("--ref", default=None, help=argparse.SUPPRESS)
    p_conc.add_argument("--ref-conc", type=float, default=None, help=argparse.SUPPRESS)
    p_conc.add_argument("--strategy", choices=["peak_max", "peak_area"], default="peak_max")
    p_conc.add_argument("--baseline", action="store_true", help="分析前执行基线校正")
    _add_baseline_args(p_conc)
    _add_tif_args(p_conc)

    # info
    p_info = subparsers.add_parser("info", help="显示光谱文件信息")
    p_info.add_argument("file", help="光谱文件路径")
    _add_tif_args(p_info)

    # tui / qt
    subparsers.add_parser("tui", help="启动终端交互界面")
    subparsers.add_parser("qt", help="启动 Qt 桌面界面")
    p_validate = subparsers.add_parser("validate-standard", help="与已知组成的标准样品/合成真值对照")
    p_validate.add_argument("manifest", help="包含真值、容差、采集条件和气体库的 JSON 清单")
    p_validate.add_argument("--report", help="输出 JSON 验证报告")
    p_validate.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(args_list)

    commands = {
        "plot": cmd_plot,
        "batch": cmd_batch,
        "snr": cmd_snr,
        "baseline": cmd_baseline,
        "concentration": cmd_concentration,
        "info": cmd_info,
        "validate-standard": cmd_validate_standard,
    }

    if args.command == "tui":
        from raman_tool.tui import main as tui_main
        return tui_main()

    if args.command == "qt":
        from raman_tool.qt_gui import main as qt_main
        return qt_main()

    if args.command in commands:
        try:
            return commands[args.command](args)
        except (ValueError, OSError, TypeError) as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
