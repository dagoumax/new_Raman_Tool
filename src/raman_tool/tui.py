"""Terminal user interface for Raman Tool."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from raman_tool.exporters import unique_path
from raman_tool.processing import calculate_concentration, calculate_snr, subtract_baseline
from raman_tool.readers import SUPPORTED_FORMATS, read_file
from raman_tool.sorting import natural_sorted
from raman_tool.visualization import plot_baseline, plot_spectrum, save_figure


console = Console()


class RamanTUI:
    def __init__(self):
        self.current_spectrum = None
        self.current_path: Path | None = None
        self.work_dir = Path.cwd()
        self.files_cache: list[Path] = []

    @property
    def current_file(self) -> str:
        return self.current_path.name if self.current_path else ""

    def run(self) -> None:
        console.show_cursor(True)
        try:
            self._main_menu()
        finally:
            console.show_cursor(True)

    def _clear(self) -> None:
        console.clear()

    def _show_banner(self) -> None:
        self._clear()
        console.print(Panel.fit(
            "[bold yellow]拉曼光谱数据处理工具[/bold yellow]\n"
            "[dim]Raman Spectroscopy Processing Tool v0.1.0[/dim]",
            border_style="cyan",
            box=ROUNDED,
        ))
        console.print(f"[dim]工作目录: {self.work_dir}[/dim]")
        if self.current_spectrum is not None:
            spec = self.current_spectrum
            console.print(
                f"[green]当前文件:[/green] {self.current_file}  "
                f"[green]点数:[/green] {spec.size}  "
                f"[green]范围:[/green] {spec.raman_shift[0]:.1f} - {spec.raman_shift[-1]:.1f}"
            )
        console.print()

    def _press_enter(self) -> None:
        console.print("\n[dim]按 Enter 返回[/dim] ", end="")
        input()

    @staticmethod
    def _fmt_size(size: int) -> str:
        if size < 1024:
            return f"{size} B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        if size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        return f"{size / (1024 * 1024 * 1024):.1f} GB"

    def _main_menu(self) -> None:
        actions = {
            "1": self._browse_files,
            "2": self._show_spectrum_detail,
            "3": self._plot_current_spectrum,
            "4": self._calc_snr,
            "5": self._baseline_correct,
            "6": self._calc_concentration,
            "7": self._batch_process_directory,
            "8": self._change_work_dir,
        }
        while True:
            self._show_banner()
            menu = Table(box=ROUNDED, show_header=False, padding=(0, 2))
            menu.add_column("", style="bold cyan", width=6)
            menu.add_column("")
            menu.add_row(" [1]", "浏览并加载文件")
            menu.add_row(" [2]", "查看光谱详情")
            menu.add_row(" [3]", "生成光谱图表")
            menu.add_row(" [4]", "计算信噪比(SNR)")
            menu.add_row(" [5]", "基线校正")
            menu.add_row(" [6]", "气体浓度分析")
            menu.add_row(" [7]", "批量处理目录")
            menu.add_row(" [8]", "切换工作目录")
            menu.add_row(" [0]", "退出")
            console.print(Panel(menu, title="[bold]主菜单[/bold]", border_style="blue", box=ROUNDED))
            choice = Prompt.ask(
                "\n[bold cyan]请选择[/bold cyan]",
                choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"],
                default="1",
            )
            if choice == "0":
                console.print("\n[green]再见![/green]")
                return
            actions[choice]()

    def _supported_files(self) -> list[Path]:
        files: list[Path] = []
        for ext in SUPPORTED_FORMATS:
            files.extend(self.work_dir.glob(f"*{ext}"))
            files.extend(self.work_dir.glob(f"*{ext.upper()}"))
        return natural_sorted(set(files))

    def _browse_files(self) -> None:
        while True:
            self._show_banner()
            self.files_cache = self._supported_files()
            if not self.files_cache:
                console.print("[bold yellow]未找到支持格式的文件[/bold yellow]")
                console.print(f"[dim]支持格式: {', '.join(SUPPORTED_FORMATS)}[/dim]")
                self._press_enter()
                return

            table = Table(box=ROUNDED, show_header=True, header_style="bold cyan")
            table.add_column("编号", width=5)
            table.add_column("文件名", style="green")
            table.add_column("格式", width=8)
            table.add_column("大小", justify="right", width=12)
            for i, file in enumerate(self.files_cache, 1):
                table.add_row(str(i), file.name, file.suffix.upper(), self._fmt_size(file.stat().st_size))
            console.print(Panel(table, title="[bold]文件列表[/bold]", border_style="blue", box=ROUNDED))

            choice = Prompt.ask("[bold cyan]输入编号，或 b 返回[/bold cyan]", default="b").strip()
            if choice.lower() == "b":
                return
            try:
                idx = int(choice) - 1
            except ValueError:
                console.print("[red]请输入有效编号[/red]")
                self._press_enter()
                continue
            if 0 <= idx < len(self.files_cache):
                self._load_file(self.files_cache[idx])
                return
            console.print("[red]编号超出范围[/red]")
            self._press_enter()

    def _load_file(self, filepath: Path) -> None:
        suffix = filepath.suffix.lower()
        kwargs = {}
        if suffix in (".tif", ".tiff", ".bmp", ".jpg", ".jpeg"):
            console.print("\n[yellow]图像导入选项，直接回车使用默认值[/yellow]")
            row_groups = Prompt.ask("  行分组，如 1-40,91-130", default="").strip()
            if row_groups:
                kwargs["row_groups"] = row_groups
            try:
                kwargs["col_merge"] = max(1, int(Prompt.ask("  列合并因子", default="1").strip()))
            except ValueError:
                kwargs["col_merge"] = 1
            if Confirm.ask("  使用行求和模式?", default=False):
                kwargs["row_mode"] = "sum"
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                task = progress.add_task(f"[cyan]正在读取 {filepath.name}...", total=None)
                self.current_spectrum = read_file(filepath, **kwargs)
                self.current_path = filepath
                self.work_dir = filepath.parent
                progress.update(task, completed=True)
            spec = self.current_spectrum
            console.print(
                f"\n[bold green]已加载 {filepath.name}[/bold green]\n"
                f"[dim]数据点数: {spec.size}, 强度范围: {spec.intensity.min():.2f} - {spec.intensity.max():.2f}[/dim]"
            )
        except Exception as exc:
            console.print(f"\n[bold red]读取失败: {exc}[/bold red]")
        self._press_enter()

    def _require_spectrum(self) -> bool:
        if self.current_spectrum is None:
            console.print("[yellow]请先加载光谱文件[/yellow]")
            self._press_enter()
            return False
        return True

    def _show_spectrum_detail(self) -> None:
        if not self._require_spectrum():
            return
        self._show_banner()
        spec = self.current_spectrum
        table = Table(box=ROUNDED, show_header=False, padding=(0, 2))
        table.add_column("属性", style="bold cyan", width=18)
        table.add_column("值")
        table.add_row("文件名", self.current_file)
        table.add_row("数据点数", str(spec.size))
        table.add_row("拉曼位移范围", f"{spec.raman_shift[0]:.2f} - {spec.raman_shift[-1]:.2f}")
        table.add_row("强度最小值", f"{spec.intensity.min():.2f}")
        table.add_row("强度最大值", f"{spec.intensity.max():.2f}")
        table.add_row("强度均值", f"{spec.intensity.mean():.2f}")
        table.add_row("强度标准差", f"{spec.intensity.std():.2f}")
        for key, value in spec.metadata.items():
            if key != "filepath":
                table.add_row(str(key), str(value))
        console.print(Panel(table, title="[bold]光谱详情[/bold]", border_style="green", box=ROUNDED))
        self._press_enter()

    def _plot_current_spectrum(self) -> None:
        if not self._require_spectrum():
            return
        spec = self.current_spectrum
        default = self.work_dir / f"{Path(self.current_file).stem}.png"
        out = unique_path(default)
        try:
            fig = plot_spectrum(spec, show=False)
            path = save_figure(fig, out)
            console.print(f"[bold green]图表已保存:[/bold green] {path}")
        except Exception as exc:
            console.print(f"[bold red]绘图失败: {exc}[/bold red]")
        self._press_enter()

    def _calc_snr(self) -> None:
        if not self._require_spectrum():
            return
        self._show_banner()
        spec = self.current_spectrum
        console.print(f"[dim]光谱范围: {spec.raman_shift[0]:.1f} - {spec.raman_shift[-1]:.1f}[/dim]")
        try:
            peak_start = peak_end = noise_start = noise_end = None
            if Confirm.ask("手动指定信号峰区域?", default=False):
                peak_start = FloatPrompt.ask("  信号起始", default=float(spec.raman_shift[0]))
                peak_end = FloatPrompt.ask("  信号结束", default=float(spec.raman_shift[-1]))
            if Confirm.ask("手动指定噪声区域?", default=False):
                noise_start = FloatPrompt.ask("  噪声起始", default=float(spec.raman_shift[0]))
                noise_end = FloatPrompt.ask("  噪声结束", default=float(spec.raman_shift[-1]))
            result = calculate_snr(spec, peak_start, peak_end, noise_start, noise_end)
            table = Table(box=ROUNDED, show_header=False, padding=(0, 2))
            table.add_row("SNR", f"{result['snr']:.2f}")
            table.add_row("信号强度", f"{result['signal']:.2f}")
            table.add_row("噪声 RMS", f"{result['noise_rms']:.4f}")
            table.add_row("峰中心", f"{result['peak_center']:.2f}")
            if result["peak_area"] > 0:
                table.add_row("峰面积", f"{result['peak_area']:.4f}")
            console.print(Panel(table, title="[bold green]计算结果[/bold green]", border_style="green", box=ROUNDED))
        except Exception as exc:
            console.print(f"[red]计算失败: {exc}[/red]")
        self._press_enter()

    def _baseline_correct(self) -> None:
        if not self._require_spectrum():
            return
        spec = self.current_spectrum
        method = Prompt.ask("基线方法", choices=["arPLS", "poly"], default="arPLS")
        try:
            if method == "arPLS":
                lam = FloatPrompt.ask("arPLS 平滑参数", default=1e5)
                corrected = subtract_baseline(spec, method="arPLS", lam=lam)
            else:
                degree = IntPrompt.ask("多项式阶数", default=3)
                corrected = subtract_baseline(spec, method="poly", degree=degree)
            baseline = spec.intensity - corrected.intensity
            out = unique_path(self.work_dir / f"{Path(self.current_file).stem}_baseline.png")
            path = save_figure(plot_baseline(spec, baseline=baseline, corrected=corrected, show=False), out)
            console.print(f"[bold green]基线图已保存:[/bold green] {path}")
            if Confirm.ask("将当前光谱替换为校正后光谱?", default=True):
                self.current_spectrum = corrected
        except Exception as exc:
            console.print(f"[red]基线校正失败: {exc}[/red]")
        self._press_enter()

    def _calc_concentration(self) -> None:
        if not self._require_spectrum():
            return
        gases = [
            ("N2", "氮气"), ("O2", "氧气"), ("CO2", "二氧化碳"),
            ("H2O", "水蒸气"), ("CH4", "甲烷"), ("H2", "氢气"),
            ("CO", "一氧化碳"), ("SO2", "二氧化硫"), ("NO", "一氧化氮"),
            ("NH3", "氨气"), ("C2H6", "乙烷"),
        ]
        table = Table(box=ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("#", width=4)
        table.add_column("代码")
        table.add_column("名称")
        for i, (code, name) in enumerate(gases, 1):
            table.add_row(str(i), code, name)
        console.print(Panel(table, title="[bold]气体浓度分析[/bold]", border_style="cyan", box=ROUNDED))
        try:
            gas_idx = IntPrompt.ask("选择目标气体", default=1) - 1
            ref_idx = IntPrompt.ask("选择参考气体", default=1) - 1
            if not (0 <= gas_idx < len(gases) and 0 <= ref_idx < len(gases)):
                raise ValueError("气体编号无效")
            ref_conc = FloatPrompt.ask("参考气体浓度(%)", default=78.0)
            window = FloatPrompt.ask("峰搜索窗口", default=10.0)
            result = calculate_concentration(
                self.current_spectrum,
                gas_name=gases[gas_idx][0],
                window=window,
                reference_gas=gases[ref_idx][0],
                reference_concentration=ref_conc,
            )
            result_table = Table(box=ROUNDED, show_header=False, padding=(0, 2))
            result_table.add_row("目标气体", result["gas"])
            result_table.add_row("浓度", f"{result['concentration']:.4f} %")
            result_table.add_row("峰中心", f"{result['peak_center']:.2f}")
            result_table.add_row("峰面积", f"{result['peak_area']:.4f}")
            result_table.add_row("参考气体", result["reference_gas"])
            result_table.add_row("参考峰", f"{result['reference_peak_center']:.2f}")
            result_table.add_row("参考面积", f"{result['reference_peak_area']:.4f}")
            console.print(Panel(result_table, title="[bold green]浓度计算结果[/bold green]", border_style="green", box=ROUNDED))
        except Exception as exc:
            console.print(f"[red]计算失败: {exc}[/red]")
        self._press_enter()

    def _batch_process_directory(self) -> None:
        files = self._supported_files()
        if not files:
            console.print("[yellow]当前目录没有支持的文件[/yellow]")
            self._press_enter()
            return
        out_dir_text = Prompt.ask("输出目录", default=str(self.work_dir / "output"))
        out_dir = Path(out_dir_text).expanduser()
        if not out_dir.is_absolute():
            out_dir = self.work_dir / out_dir
        do_baseline = Confirm.ask("执行基线校正?", default=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        ok = fail = 0
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("[cyan]批量处理中...", total=None)
            for file in files:
                try:
                    spec = read_file(file)
                    if do_baseline:
                        spec = subtract_baseline(spec)
                    out = unique_path(out_dir / f"{file.stem}.png")
                    save_figure(plot_spectrum(spec, show=False), out)
                    ok += 1
                except Exception:
                    fail += 1
            progress.update(task, completed=True)
        console.print(f"[green]完成: {ok} 成功[/green], [red]{fail} 失败[/red], 输出目录: {out_dir}")
        self._press_enter()

    def _change_work_dir(self) -> None:
        value = Prompt.ask("新的工作目录", default=str(self.work_dir)).strip()
        path = Path(value).expanduser()
        if path.is_dir():
            self.work_dir = path
            console.print(f"[green]工作目录已切换到 {path}[/green]")
        else:
            console.print(f"[red]目录不存在: {path}[/red]")
        self._press_enter()


def main() -> int:
    try:
        RamanTUI().run()
    except KeyboardInterrupt:
        console.print("\n[green]Goodbye![/green]")
    except Exception as exc:
        console.print(f"\n[red]Error: {exc}[/red]")
        return 1
    finally:
        console.show_cursor(True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
