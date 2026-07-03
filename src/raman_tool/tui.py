"""终端用户界面 (Terminal UI).

基于 rich 库的交互式终端界面。支持菜单导航、文件浏览、光谱查看、信噪比计算、气体浓度分析等功能。"""
import os, sys; from pathlib import Path
from raman_tool.models import Spectrum
from raman_tool.readers import read_file, detect_format, SUPPORTED_FORMATS
from raman_tool.processing import calculate_snr, calculate_concentration, subtract_baseline
from raman_tool.visualization import plot_spectrum, plot_baseline, save_figure
from rich.console import Console; from rich.table import Table
from rich.panel import Panel; from rich.text import Text
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.layout import Layout; from rich.live import Live
from rich.box import ROUNDED, HEAVY
console = Console()

class RamanTUI:
    def __init__(self):
        self.current_spectrum = None
        self.current_file = ""
        self.work_dir = Path.cwd()
        self.files_cache = []
    def run(self):
        console.show_cursor(True)
        self._show_banner()
        try:
            self._main_menu()
        finally:
            console.show_cursor(True)
    def _clear(self):
        os.system("cls" if sys.platform == "win32" else "clear")
    def _show_banner(self):
        self._clear()
        banner = r"""
  [bold cyan]╔══════════════════════════════════════════════════╗[/bold cyan]
  ║                                                 ║
  ║  [bold yellow]拉曼光谱数据处理工具[/bold yellow]                               ║
  ║  [dim]Raman Spectroscopy Processing Tool v0.1.0[/dim]          ║
  ║                                                 ║
  [bold cyan]╚══════════════════════════════════════════════════╝[/bold cyan]
"""
        console.print(banner)
        console.print(f"  [dim]工作目录: {self.work_dir}[/dim]\n")
    def _press_enter(self):
        """等待用户按 Enter 返回，确保光标可见。"""
        console.show_cursor(True)
        console.print("\n[dim]按Enter返回[/dim] ", end="")
        try:
            input()
        finally:
            console.show_cursor(True)
    def _main_menu(self):
        while True:
            self._clear()
            self._show_banner()
            if self.current_spectrum:
                console.print(Panel(
                    f"[green]当前文件:[/green] {self.current_file}\n"
                    f"[green]数据点数:[/green] {self.current_spectrum.size}\n"
                    f"[green]拉曼位移范围:[/green] {self.current_spectrum.raman_shift[0]:.1f} - "
                    f"{self.current_spectrum.raman_shift[-1]:.1f} cm-1\n"
                    f"[green]强度范围:[/green] {self.current_spectrum.intensity.min():.1f} - "
                    f"{self.current_spectrum.intensity.max():.1f}",
                    title="[bold]当前光谱[/bold]",
                    border_style="green", box=ROUNDED,
                ))
            menu = Table(box=ROUNDED, show_header=False, padding=(0, 2))
            menu.add_column("", style="bold cyan", width=6)
            menu.add_column("")
            menu.add_row(" [1]", "浏览并载入文件")
            menu.add_row(" [2]", "查看光谱数据详情")
            menu.add_row(" [3]", "生成光谱图表")
            menu.add_row(" [4]", "计算信噪比(SNR)")
            menu.add_row(" [5]", "基线校正")
            menu.add_row(" [6]", "气体浓度分析")
            menu.add_row(" [7]", "批量处理目录")
            menu.add_row(" [8]", "切换工作目录")
            menu.add_row(" [0]", "退出")
            console.print(Panel(menu, title="[bold]主菜单[/bold]", border_style="blue", box=ROUNDED))
            choice = Prompt.ask("\n[bold cyan]请选择[/bold cyan]", choices=["0","1","2","3","4","5","6","7","8"], default="1")
            if choice == "0":
                console.print("\n[green]再见![/green]")
                console.show_cursor(True)
                break
    @staticmethod
    def _fmt_size(size):
        if size < 1024: return f"{size} B"
        elif size < 1024 * 1024: return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024: return f"{size / (1024 * 1024):.1f} MB"
        else: return f"{size / (1024 * 1024 * 1024):.1f} GB"

    def _browse_files(self):
        console.print("Browse files - TBD")

    def _load_file(self, filepath):
        self.current_spectrum = read_file(filepath)
        self.current_file = filepath.name
        self.work_dir = filepath.parent
        console.print(f"[green]Loaded {filepath.name}[/green]")

    def _show_spectrum_detail(self):
        console.print("[yellow]Spectrum detail - TBD[/yellow]")

    def _calc_snr(self):
        console.print("[yellow]SNR calculation - TBD[/yellow]")

    def _calc_concentration(self):
        console.print("[yellow]Concentration analysis - TBD[/yellow]")


    def _browse_files(self):
        while True:
            self._clear(); self._show_banner()
            supported = set(SUPPORTED_FORMATS.keys())
            self.files_cache = []
            for ext in supported:
                self.files_cache.extend(sorted(self.work_dir.glob(f"*{ext}")))
                self.files_cache.extend(sorted(self.work_dir.glob(f"*{ext.upper()}")))
            if not self.files_cache:
                console.print(f"\n[bold yellow]未找到支持格式的文件[/bold yellow]")
                console.print(f"[dim]支持的格式: {', '.join(supported)}[/dim]")
                console.print(f"[dim]工作目录: {self.work_dir}[/dim]")
                self._press_enter(); return
            file_table = Table(box=ROUNDED, show_header=True, header_style="bold cyan")
            file_table.add_column("编号", style="dim", width=5)
            file_table.add_column("文件名", style="green")
            file_table.add_column("格式", style="yellow", width=8)
            file_table.add_column("大小", style="dim", width=12, justify="right")
            for i, f in enumerate(self.files_cache, 1):
                size_str = self._fmt_size(f.stat().st_size) if f.exists() else "?"
                file_table.add_row(str(i), f.name, f.suffix.upper(), size_str)
            console.print(Panel(file_table, title="[bold]文件列表[/bold]", border_style="blue", box=ROUNDED))
            console.print("\n[dim]输入编号, [/dim][dim cyan]搜索[/dim cyan][dim] 搜索, [/dim][dim cyan]返回[/dim cyan][dim] 返回[/dim]")
            choice = Prompt.ask("[bold cyan]选择[/bold cyan]").strip()
            if choice.lower() == "b": return
            elif choice.lower() == "s":
                query = Prompt.ask("[bold]Search file name[/bold]").strip().lower()
                self.files_cache = [f for f in self.files_cache if query in f.name.lower()]; continue
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(self.files_cache):
                    self._load_file(self.files_cache[idx]); return
                else: console.print("[red]Invalid number[/red]"); self._press_enter()
            except ValueError: console.print("[red]Please enter a number[/red]"); self._press_enter()

    def _load_file(self, filepath):
        suffix = filepath.suffix.lower()
        kw = {}
        if suffix in (".tif", ".tiff", ".bmp"):
            console.print("\n[yellow]TIF/BMP 文件 -> 导入选项 (回车跳过)[/yellow]")
            rg = Prompt.ask("  行分组 (如1-40,91-130)", default="").strip()
            if rg: kw["row_groups"] = rg
            cm = Prompt.ask("  列合并因子", default="1").strip()
            try: kw["col_merge"] = max(1, int(cm))
            except: kw["col_merge"] = 1
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                task = progress.add_task(f"[cyan]正在读取 {{filepath.name}}...", total=None)
                self.current_spectrum = read_file(filepath, **kw)
                self.current_file = filepath.name
                self.work_dir = filepath.parent
                progress.update(task, completed=True)
            console.print(f"\n[bold green]✓ 已加载 {{filepath.name}}[/bold green]")
            console.print(f"  [dim]数据点数: {self.current_spectrum.size}, 强度范围: {self.current_spectrum.intensity.min():.0f} - {self.current_spectrum.intensity.max():.0f}[/dim]")
        except Exception as e:
            console.print(f"\n[bold red]✗ 读取失败: {{e}}[/bold red]")
        self._press_enter()

    def _show_spectrum_detail(self):
        if self.current_spectrum is None:
            console.print("[yellow]请先加载光谱文件[/yellow]")
            self._press_enter(); return
        self._clear(); self._show_banner()
        spec = self.current_spectrum
        info_table = Table(box=ROUNDED, show_header=False, padding=(0, 2))
        info_table.add_column("属性", style="bold cyan", width=16)
        info_table.add_column("值")
        info_table.add_row("文件名", self.current_file)
        info_table.add_row("数据点数", str(spec.size))
        info_table.add_row("拉曼位移范围", f"{spec.raman_shift[0]:.2f} - {spec.raman_shift[-1]:.2f} cm-1")
        info_table.add_row("强度最小值", f"{spec.intensity.min():.2f}")
        info_table.add_row("强度最大值", f"{spec.intensity.max():.2f}")
        info_table.add_row("强度平均值", f"{spec.intensity.mean():.2f}")
        info_table.add_row("强度标准差", f"{spec.intensity.std():.2f}")
        for key, val in spec.metadata.items():
            if key != "filepath": info_table.add_row(str(key), str(val))
        console.print(Panel(info_table, title="[bold]光谱详情[/bold]", border_style="green", box=ROUNDED))
        data_table = Table(box=ROUNDED, show_header=True, header_style="bold")
        data_table.add_column("Pixel", style="dim", width=8)
        data_table.add_column("Raman Shift", width=14)
        data_table.add_column("Intensity", width=14)
        n = min(20, spec.size)
        for i in range(n):
            data_table.add_row(str(i), f"{spec.raman_shift[i]:.2f}", f"{spec.intensity[i]:.2f}")
        if spec.size > 20: data_table.add_row("...", "...", "...")
        console.print(Panel(data_table, title=f"[bold]前 {n} 个数据点[/bold]", border_style="blue", box=ROUNDED))
        self._press_enter()

    def _calc_snr(self):
        if self.current_spectrum is None:
            console.print("[yellow]请先加载光谱文件[/yellow]")
            self._press_enter(); return
        self._clear(); self._show_banner()
        spec = self.current_spectrum
        console.print(Panel(f"[dim]光谱范围: {spec.raman_shift[0]:.1f} - {spec.raman_shift[-1]:.1f} cm-1[/dim]\n"
            f"[yellow]请输入信号峰区域和噪声区域[/yellow]",
            title="[bold]信噪比计算[/bold]",
            border_style="cyan", box=ROUNDED))
        try:
            use_peak = Confirm.ask("\n手动指定信号峰区域?", default=False)
            peak_start = None; peak_end = None
            if use_peak:
                peak_start = FloatPrompt.ask("  信号起始 (cm-1)", default=0.0)
                peak_end = FloatPrompt.ask("  信号结束 (cm-1)", default=float(spec.size/4))
            use_noise = Confirm.ask("手动指定噪声区域?", default=False)
            noise_start = None; noise_end = None
            if use_noise:
                noise_start = FloatPrompt.ask("  噪声起始 (cm-1)", default=float(spec.size*0.8))
                noise_end = FloatPrompt.ask("  噪声结束 (cm-1)", default=float(spec.size-1))
            result = calculate_snr(spec, peak_start, peak_end, noise_start, noise_end)
            result_table = Table(box=ROUNDED, show_header=False, padding=(0, 2))
            result_table.add_column("", style="bold cyan", width=14)
            result_table.add_column("")
            snr_style = "green" if result["snr"] < float("inf") else "yellow"
            result_table.add_row("信噪比 SNR", f"[bold {snr_style}]{result['snr']:.2f}[/bold {snr_style}]")
            result_table.add_row("信号强度", f"{result['signal']:.2f}")
            result_table.add_row("噪声 RMS", f"{result['noise_rms']:.4f}")
            result_table.add_row("峰中心", f"{result['peak_center']:.2f} cm-1")
            if result["peak_area"] > 0: result_table.add_row("峰面积", f"{result['peak_area']:.4f}")
            console.print(Panel(result_table, title="[bold green]计算结果[/bold green]", border_style="green", box=ROUNDED))
        except (ValueError, TypeError) as e:
            console.print(f"[red]输入错误: {e}[/red]")
        self._press_enter()

    def _calc_concentration(self):
        if self.current_spectrum is None:
            console.print("[yellow]请先加载光谱文件[/yellow]")
            self._press_enter(); return
        self._clear(); self._show_banner()
        spec = self.current_spectrum
        gases = [
            ("N2", 氮气), ("O2", 氧气),
            ("CO2", 二氧化碳),
            ("H2O", 水蒸气),
            ("CH4", 甲烷), ("H2", 氢气),
            ("CO", 一氧化碳),
            ("SO2", 二氧化硫),
            ("NO", 一氧化氮),
            ("NH3", 氨气),
            ("C2H6", 乙烷),
        ]
        gas_table = Table(box=ROUNDED, show_header=True, header_style="bold cyan")
        gas_table.add_column("#", style="dim", width=4)
        gas_table.add_column("代码")
        gas_table.add_column("名称")
        for i, (code, name) in enumerate(gases, 1):
            gas_table.add_row(str(i), code, name)
        console.print(Panel(gas_table, title="[bold]气体浓度分析[/bold]", border_style="cyan", box=ROUNDED))
        try:
            gas_idx = IntPrompt.ask("\n[bold cyan]选择目标气体[/bold cyan]", default=1) - 1
            if not (0 <= gas_idx < len(gases)):
                console.print("[red]无效选择[/red]"); self._press_enter(); return
            gas_name = gases[gas_idx][0]
            ref_idx = IntPrompt.ask("[bold cyan]选择参考气体[/bold cyan]", default=1) - 1
            if not (0 <= ref_idx < len(gases)):
                console.print("[red]无效选择[/red]"); self._press_enter(); return
            ref_gas = gases[ref_idx][0]
            ref_conc = FloatPrompt.ask("参考气体浓度(%)", default=78.0)
            window = FloatPrompt.ask("峰搜索窗口(cm-1)", default=10.0)
            result = calculate_concentration(spec, gas_name=gas_name, window=window, reference_gas=ref_gas, reference_concentration=ref_conc)
            result_table = Table(box=ROUNDED, show_header=False, padding=(0, 2))
            result_table.add_row("目标气体", result["gas"])
            result_table.add_row("浓度", f"[bold green]{result['concentration']:.4f} %[/bold green]")
            result_table.add_row("峰中心", f"{result['peak_center']:.2f} cm-1")
            result_table.add_row("峰面积", f"{result['peak_area']:.4f}")
            result_table.add_row("参考气体", result["reference_gas"])
            result_table.add_row("参考峰", f"{result['reference_peak_center']:.2f} cm-1")
            result_table.add_row("参考面积", f"{result['reference_peak_area']:.4f}")
            console.print(Panel(result_table, title="[bold green]浓度计算结果[/bold green]", border_style="green", box=ROUNDED))
        except (ValueError, TypeError) as e:
            console.print(f"[red]错误: {e}[/red]")
        self._press_enter()


def main():
    try:
        console.show_cursor(True)
        app = RamanTUI()
        app.run()
    except KeyboardInterrupt:
        console.show_cursor(True)
        console.print("\n[green]Goodbye![/green]")
    except Exception as e:
        console.show_cursor(True)
        console.print(f"\n[red]Error: {e}[/red]")
        return 1
    finally:
        console.show_cursor(True)
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
