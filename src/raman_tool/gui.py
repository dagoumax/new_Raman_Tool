"""鍥惧舰鐢ㄦ埛鐣岄潰 (GUI).

鍩轰簬 tkinter锛屾敮鎸佹嫋鎷藉鍏ャ€佸厜璋辨樉绀恒€佷俊鍣瘮璁＄畻銆佹皵浣撴祿搴﹀垎鏋愩€?"""

import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from raman_tool.models import Spectrum
from raman_tool.readers import read_file, SUPPORTED_FORMATS, detect_format
from raman_tool.processing import calculate_snr, calculate_concentration, subtract_baseline
from raman_tool.visualization import plot_spectrum, plot_baseline, save_figure
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# 灏濊瘯瀵煎叆鎷栨嫿鏀寔
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except ImportError:
    HAS_DND = False


class RamanToolGUI:
    """鎷夋浖鍏夎氨宸ュ叿 GUI 涓荤獥鍙?"""

    def __init__(self):
        if HAS_DND:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()

        self.root.title("鎷夋浖鍏夎氨鏁版嵁澶勭悊宸ュ叿")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)

        self.current_spectrum: Spectrum | None = None
        self.current_file: str = ""
        self.current_figure: plt.Figure | None = None
        self.spectra_cache: list[Spectrum] = []

        self._setup_ui()
        if HAS_DND:
            self._setup_drag_drop()

    def _setup_ui(self):
        """鍒濆鍖栫晫闈㈠竷灞€."""
        # 宸︿晶鎺у埗闈㈡澘
        control_frame = ttk.Frame(self.root, width=320, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        control_frame.pack_propagate(False)

        # 鏍囬
        title_label = ttk.Label(
            control_frame, text="鎷夋浖鍏夎氨宸ュ叿", font=("Microsoft YaHei UI", 14, "bold")
        )
        title_label.pack(pady=(0, 15))

        # 瀵煎叆鍖哄煙
        import_frame = ttk.LabelFrame(control_frame, text="鏂囦欢瀵煎叆", padding=10)
        import_frame.pack(fill=tk.X, pady=(0, 10))

        if HAS_DND:
            drop_hint = "鎷栨嫿鏂囦欢鍒扮獥鍙?鎴?
            drop_area = tk.Label(
                import_frame,
                text="鏂囦欢鎷栨嫿鍖哄煙\n\n灏嗗厜璋辨枃浠舵嫋鎷藉埌姝ゅ\n鎴栧彸渚х粯鍥惧尯鍩?,
                bg="#e8f0fe",
                fg="#3a3a3a",
                relief=tk.GROOVE,
                height=4,
                font=("Microsoft YaHei UI", 9),
            )
            drop_area.pack(fill=tk.X, pady=(0, 8), ipady=8)

        self._file_listbox = tk.Listbox(
            import_frame, height=6, selectmode=tk.EXTENDED
        )
        self._file_listbox.pack(fill=tk.X, pady=(0, 8))
        self._file_listbox.bind("<<ListboxSelect>>", self._on_file_select)

        btn_frame = ttk.Frame(import_frame)
        btn_frame.pack(fill=tk.X)

        btn_add = ttk.Button(btn_frame, text="娣诲姞鏂囦欢", command=self._add_files)
        btn_add.pack(side=tk.LEFT, padx=(0, 5))

        btn_clear = ttk.Button(btn_frame, text="娓呴櫎", command=self._clear_files)
        btn_clear.pack(side=tk.LEFT)

        # 澶勭悊鍖哄煙
        process_frame = ttk.LabelFrame(control_frame, text="鏁版嵁鎿嶄綔", padding=10)
        process_frame.pack(fill=tk.X, pady=(0, 10))

        btn_refresh = ttk.Button(process_frame, text="鍒锋柊鍥捐〃", command=self._refresh_plot)
        btn_refresh.pack(fill=tk.X, pady=(0, 5))

        btn_baseline = ttk.Button(process_frame, text="鍩虹嚎鏍℃", command=self._baseline_correct)
        btn_baseline.pack(fill=tk.X, pady=(0, 5))

        btn_export = ttk.Button(process_frame, text="瀵煎嚭鍥剧墖", command=self._export_figure)
        btn_export.pack(fill=tk.X, pady=(0, 5))

        # 淇″櫔姣斿尯鍩?        snr_frame = ttk.LabelFrame(control_frame, text="淇″櫔姣旇绠?, padding=10)
        snr_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(snr_frame, text="淇″彿鍖哄煙 (cm-1):").pack(anchor=tk.W)
        snr_peak_frame = ttk.Frame(snr_frame)
        snr_peak_frame.pack(fill=tk.X, pady=(2, 5))
        self._snr_peak_start = ttk.Entry(snr_peak_frame, width=10)
        self._snr_peak_start.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(snr_peak_frame, text="-").pack(side=tk.LEFT, padx=(0, 5))
        self._snr_peak_end = ttk.Entry(snr_peak_frame, width=10)
        self._snr_peak_end.pack(side=tk.LEFT)

        ttk.Label(snr_frame, text="鍣０鍖哄煙 (cm-1):").pack(anchor=tk.W)
        snr_noise_frame = ttk.Frame(snr_frame)
        snr_noise_frame.pack(fill=tk.X, pady=(2, 5))
        self._snr_noise_start = ttk.Entry(snr_noise_frame, width=10)
        self._snr_noise_start.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(snr_noise_frame, text="-").pack(side=tk.LEFT, padx=(0, 5))
        self._snr_noise_end = ttk.Entry(snr_noise_frame, width=10)
        self._snr_noise_end.pack(side=tk.LEFT)

        btn_snr = ttk.Button(snr_frame, text="璁＄畻淇″櫔姣?, command=self._calc_snr)
        btn_snr.pack(fill=tk.X, pady=(5, 0))

        # 娴撳害璁＄畻鍖哄煙
        conc_frame = ttk.LabelFrame(control_frame, text="姘斾綋娴撳害", padding=10)
        conc_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(conc_frame, text="鐩爣姘斾綋:").pack(anchor=tk.W)
        self._gas_var = tk.StringVar(value="N2")
        gas_combo = ttk.Combobox(conc_frame, textvariable=self._gas_var, state="readonly")
        gas_combo["values"] = ["N2", "O2", "CO2", "H2O", "CH4", "H2", "CO", "SO2", "NO", "NH3", "C2H6"]
        gas_combo.pack(fill=tk.X, pady=(2, 5))

        ttk.Label(conc_frame, text="鍙傝€冩皵浣?").pack(anchor=tk.W)
        self._ref_var = tk.StringVar(value="N2")
        ref_combo = ttk.Combobox(conc_frame, textvariable=self._ref_var, state="readonly")
        ref_combo["values"] = ["N2", "O2", "CO2", "H2O", "CH4", "H2", "CO", "SO2", "NO", "NH3", "C2H6"]
        ref_combo.pack(fill=tk.X, pady=(2, 5))

        ttk.Label(conc_frame, text="鍙傝€冩祿搴?(%):").pack(anchor=tk.W)
        self._ref_conc_var = tk.StringVar(value="78.0")
        ref_conc_entry = ttk.Entry(conc_frame, textvariable=self._ref_conc_var, width=10)
        ref_conc_entry.pack(fill=tk.X, pady=(2, 5))

        btn_conc = ttk.Button(conc_frame, text="璁＄畻娴撳害", command=self._calc_concentration)
        btn_conc.pack(fill=tk.X, pady=(5, 0))

        # 缁撴灉鏄剧ず
        result_frame = ttk.LabelFrame(control_frame, text="缁撴灉杈撳嚭", padding=10)
        result_frame.pack(fill=tk.X, fill=tk.Y, expand=True)

        self._result_text = tk.Text(result_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        self._result_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self._result_text, command=self._result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._result_text.configure(yscrollcommand=scrollbar.set)

        # 鐘舵€佹爮
        self._status_var = tk.StringVar(value="灏辩华")
        status_bar = ttk.Label(
            self.root,
            textvariable=self._status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2),
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 鍙充晶缁樺浘鍖哄煙
        plot_frame = ttk.Frame(self.root, padding=5)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._plot_frame = plot_frame

    def _setup_drag_drop(self):
        """璁剧疆鎷栨嫿鏀寔."""
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop(self, event):
        """澶勭悊鎷栨嫿鏂囦欢."""
        files = self._parse_drop_data(event.data)
        self._load_files(files)

    def _parse_drop_data(self, data: str) -> list[str]:
        """瑙ｆ瀽鎷栨嫿鏁版嵁涓殑鏂囦欢璺緞.

        Windows 鏍煎紡: {path1} {path2} ...
        Linux 鏍煎紡: path1\npath2\n...
        """
        if "{" in data:
            import re
            files = re.findall(r"\{(.+?)\}", data)
        else:
            files = data.split()

        valid_files = []
        for f in files:
            f = f.strip()
            if not f:
                continue
            try:
                suffix = detect_format(f)
                valid_files.append(f)
            except ValueError:
                pass

        return valid_files

    def _add_files(self):
        """閫氳繃鏂囦欢瀵硅瘽妗嗘坊鍔犳枃浠?"""
        patterns = [
            ("鎵€鏈夋敮鎸佹牸寮?, "*.txt;*.asc;*.sif;*.tif;*.tiff;*.bmp"),
            ("TXT 鏂囦欢", "*.txt"),
            ("ASC 鏂囦欢", "*.asc"),
            ("SIF 鏂囦欢", "*.sif"),
            ("TIFF 鏂囦欢", "*.tif;*.tiff"),
            ("BMP 鏂囦欢", "*.bmp"),
            ("鎵€鏈夋枃浠?, "*.*"),
        ]

        files = filedialog.askopenfilenames(
            title="閫夋嫨鍏夎氨鏂囦欢",
            filetypes=patterns,
        )
        if files:
            self._load_files(files)

    def _load_files(self, files: list[str]):
        """鍔犺浇鏂囦欢鍒楄〃."""
        added = 0
        for filepath in files:
            if filepath in self._file_listbox.get(0, tk.END):
                continue
            self._file_listbox.insert(tk.END, filepath)
            added += 1

        if added > 0:
            self._status_var.set(f"宸叉坊鍔?{added} 涓枃浠?)
            if self._file_listbox.size() == 1:
                self._file_listbox.selection_set(0)
                self._on_file_select()

    def _clear_files(self):
        """娓呴櫎鏂囦欢鍒楄〃."""
        self._file_listbox.delete(0, tk.END)
        self.spectra_cache.clear()
        self.current_spectrum = None
        self.current_file = ""
        self._clear_plot()
        self._status_var.set("宸叉竻闄ゆ墍鏈夋枃浠?)

    def _on_file_select(self, event=None):
        """鏂囦欢閫夋嫨鍙樺寲澶勭悊."""
        selection = self._file_listbox.curselection()
        if not selection:
            return

        selected_files = [self._file_listbox.get(i) for i in selection]

        # 鍔犺浇閫変腑鐨勫厜璋?        self.spectra_cache.clear()
        for filepath in selected_files:
            try:
                spectrum = read_file(Path(filepath))
                self.spectra_cache.append(spectrum)
            except Exception as e:
                self._append_result(f"[閿欒] {filepath}: {e}\n")

        if self.spectra_cache:
            self.current_spectrum = self.spectra_cache[0]
            self.current_file = selected_files[0]
            self._append_result(f"宸插姞杞? {Path(self.current_file).name}\n")
            self._append_result(
                f"  鏁版嵁鐐规暟: {self.current_spectrum.size}, "
                f"鑼冨洿: {self.current_spectrum.raman_shift[0]:.1f} - "
                f"{self.current_spectrum.raman_shift[-1]:.1f} cm-1\n\n"
            )
            self._refresh_plot()

    def _refresh_plot(self):
        """鍒锋柊鍥捐〃闈㈡澘."""
        if not self.spectra_cache:
            return

        self._clear_plot()

        from raman_tool.visualization import plot_multiple, plot_spectrum

        if len(self.spectra_cache) > 1:
            fig = plot_multiple(self.spectra_cache, show=False)
        else:
            fig = plot_spectrum(self.spectra_cache[0], show=False)

        self.current_figure = fig
        canvas = FigureCanvasTkAgg(fig, master=self._plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas = canvas

    def _clear_plot(self):
        """娓呯┖缁樺浘鍖哄煙."""
        for widget in self._plot_frame.winfo_children():
            widget.destroy()
        self._canvas = None

    def _baseline_correct(self):
        """鎵ц鍩虹嚎鏍℃."""
        if self.current_spectrum is None:
            messagebox.showwarning("鎻愮ず", "璇峰厛鍔犺浇鍏夎氨鏂囦欢")
            return

        try:
            degree = 3
            corrected = subtract_baseline(self.current_spectrum, "arPLS")
            baseline = self.current_spectrum.intensity - corrected.intensity

            self._clear_plot()
            fig = plot_baseline(
                self.current_spectrum,
                baseline=baseline,
                corrected=corrected,
                show=False,
            )
            self.current_figure = fig
            canvas = FigureCanvasTkAgg(fig, master=self._plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._canvas = canvas

            self._append_result("鍩虹嚎鏍℃瀹屾垚 (澶氶」寮? deg=3)\n")
        except Exception as e:
            self._append_result(f"[閿欒] 鍩虹嚎鏍℃澶辫触: {e}\n")

    def _export_figure(self):
        """瀵煎嚭褰撳墠鍥捐〃."""
        if self.current_figure is None:
            messagebox.showwarning("鎻愮ず", "璇峰厛鐢熸垚鍥捐〃")
            return

        filepath = filedialog.asksaveasfilename(
            title="淇濆瓨鍥剧墖",
            defaultextension=".png",
            filetypes=[
                ("PNG 鍥剧墖", "*.png"),
                ("PDF 鏂囨。", "*.pdf"),
                ("SVG 鐭㈤噺鍥?, "*.svg"),
            ],
        )

        if filepath:
            try:
                save_figure(self.current_figure, filepath)
                self._status_var.set(f"宸蹭繚瀛? {filepath}")
                self._append_result(f"鍥捐〃宸插鍑? {filepath}\n")
            except Exception as e:
                messagebox.showerror("閿欒", f"淇濆瓨澶辫触: {e}")

    def _calc_snr(self):
        """璁＄畻淇″櫔姣?"""
        if self.current_spectrum is None:
            messagebox.showwarning("鎻愮ず", "璇峰厛鍔犺浇鍏夎氨鏂囦欢")
            return

        try:
            peak_start = None
            peak_end = None
            noise_start = None
            noise_end = None

            ps = self._snr_peak_start.get().strip()
            pe = self._snr_peak_end.get().strip()
            if ps and pe:
                peak_start = float(ps)
                peak_end = float(pe)

            ns = self._snr_noise_start.get().strip()
            ne = self._snr_noise_end.get().strip()
            if ns and ne:
                noise_start = float(ns)
                noise_end = float(ne)

            result = calculate_snr(
                self.current_spectrum, peak_start, peak_end, noise_start, noise_end
            )

            self._append_result("=== 淇″櫔姣旇绠?===\n")
            self._append_result(f"SNR:       {result['snr']:.2f}\n")
            self._append_result(f"淇″彿寮哄害:  {result['signal']:.2f}\n")
            self._append_result(f"鍣０ RMS:  {result['noise_rms']:.4f}\n")
            self._append_result(f"宄颁腑蹇?    {result['peak_center']:.2f} cm-1\n")
            if result["peak_area"] > 0:
                self._append_result(f"宄伴潰绉?    {result['peak_area']:.4f}\n")
            self._append_result("\n")
        except (ValueError, TypeError) as e:
            self._append_result(f"[閿欒] 璇锋鏌ヨ緭鍏ュ弬鏁? {e}\n")

    def _calc_concentration(self):
        """璁＄畻姘斾綋娴撳害."""
        if self.current_spectrum is None:
            messagebox.showwarning("鎻愮ず", "璇峰厛鍔犺浇鍏夎氨鏂囦欢")
            return

        gas = self._gas_var.get()
        ref = self._ref_var.get()

        try:
            ref_conc = float(self._ref_conc_var.get())
        except ValueError:
            self._append_result("[閿欒] 鍙傝€冩祿搴﹀繀椤绘槸鏁板瓧\n")
            return

        try:
            result = calculate_concentration(
                self.current_spectrum,
                gas_name=gas,
                window=10.0,
                reference_gas=ref,
                reference_concentration=ref_conc,
            )

            self._append_result("=== 姘斾綋娴撳害璁＄畻 ===\n")
            self._append_result(f"姘斾綋:      {result['gas']}\n")
            self._append_result(f"娴撳害:      {result['concentration']:.4f} %\n")
            self._append_result(f"宄颁腑蹇?    {result['peak_center']:.2f} cm-1\n")
            self._append_result(f"宄伴潰绉?    {result['peak_area']:.4f}\n")
            self._append_result(f"鍙傝€冩皵浣?  {result['reference_gas']}\n")
            self._append_result(f"鍙傝€冩祿搴?  {ref_conc:.1f} %\n")
            self._append_result(f"鍙傝€冨嘲涓?  {result['reference_peak_center']:.2f} cm-1\n")
            self._append_result(f"鍙傝€冮潰绉?  {result['reference_peak_area']:.4f}\n")
            self._append_result("\n")
        except Exception as e:
            self._append_result(f"[閿欒] 娴撳害璁＄畻澶辫触: {e}\n")

    def _append_result(self, text: str):
        """杩藉姞缁撴灉鏂囨湰."""
        self._result_text.configure(state=tk.NORMAL)
        self._result_text.insert(tk.END, text)
        self._result_text.see(tk.END)
        self._result_text.configure(state=tk.DISABLED)

    def run(self):
        """杩愯 GUI 涓诲惊鐜?"""
        self.root.mainloop()


def main() -> int:
    """GUI 鍏ュ彛."""
    app = RamanToolGUI()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

