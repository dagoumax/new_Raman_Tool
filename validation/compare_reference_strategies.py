"""Compare fixed default strategies on independently identified real samples.

No coefficients, windows or baseline settings are fitted to these samples.
The pure-CO2 comparison uses the user's 100% reference and 1 percentage-point
tolerance. Air values remain signal fractions until response calibration.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from raman_tool.gas_library import DEFAULT_GAS_LIBRARY
from raman_tool.processing.baseline import subtract_baseline
from raman_tool.processing.concentration import calculate_gas_concentrations
from raman_tool.readers import read_file, SUPPORTED_FORMATS


SETTINGS = {"method": "arPLS", "lam": 1e5, "max_iter": 50, "tol": 1e-6}


def compare(directory: Path, label: str):
    records = []
    files = sorted(path for path in directory.rglob("*")
                   if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS)
    for path in files:
        record = {"group": label, "path": str(path.resolve()),
                  "relative_path": str(path.relative_to(directory)),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "results": {}}
        try:
            raw = read_file(path)
            record["source_metadata"] = {
                "points": raw.size, "x_unit": raw.x_unit,
                "x_range": [float(raw.raman_shift.min()), float(raw.raman_shift.max())],
                **{key: raw.metadata.get(key) for key in (
                    "calibration_source", "calibration_coefficients", "laser_wavelength",
                    "data_format", "frame_data_offset", "frame_data_values", "image_shape",
                )},
            }
            spectra = {"raw": raw, "arPLS": subtract_baseline(raw, **SETTINGS)}
            for preprocessing, spectrum in spectra.items():
                for strategy in ("peak_max", "peak_area"):
                    key = f"{preprocessing}_{strategy}"
                    result = calculate_gas_concentrations(spectrum, strategy=strategy, library=DEFAULT_GAS_LIBRARY)
                    record["results"][key] = result
                    if label == "user_identified_pure_CO2":
                        errors = {gas: abs(result["percentages"][gas] - expected)
                                  for gas, expected in {"CO2": 100.0, "N2": 0.0, "O2": 0.0}.items()}
                        error = errors["CO2"]
                        result.update(reference_co2_percent=100.0, tolerance_percentage_points=1.0,
                                      absolute_error_percentage_points=error, meets_requested_co2_tolerance=error <= 1.0)
                        result["reference_comparison"] = {
                            "expected_percentages": {"CO2": 100.0, "N2": 0.0, "O2": 0.0},
                            "absolute_errors_percentage_points": errors,
                            "passed_by_gas": {gas: value <= 1.0 for gas, value in errors.items()},
                            "all_gases_within_1_percentage_point": all(value <= 1.0 for value in errors.values()),
                        }
        except Exception as exc:
            record.update(error_type=type(exc).__name__, error=str(exc))
        records.append(record)
    summary = {"samples": len(records), "errors": sum("error" in row for row in records), "strategies": {}}
    for key in ("raw_peak_max", "raw_peak_area", "arPLS_peak_max", "arPLS_peak_area"):
        strategy_summary = {}
        for gas in ("CO2", "N2", "O2"):
            values = [row["results"][key]["percentages"][gas] for row in records if key in row["results"]]
            if values:
                strategy_summary[gas] = {"min": min(values), "max": max(values),
                                         "mean": float(np.mean(values)), "std": float(np.std(values))}
                centers = [row["results"][key]["peaks"][gas]["center"] for row in records if key in row["results"]]
                strategy_summary[gas]["peak_center_cm1"] = {
                    "min": min(centers), "max": max(centers), "mean": float(np.mean(centers)),
                    "std": float(np.std(centers)),
                }
        if label == "user_identified_pure_CO2":
            strategy_summary["co2_within_1_percentage_point"] = sum(
                row["results"].get(key, {}).get("meets_requested_co2_tolerance", False) for row in records)
            strategy_summary["all_gases_within_1_percentage_point"] = sum(
                row["results"].get(key, {}).get("reference_comparison", {}).get("all_gases_within_1_percentage_point", False)
                for row in records)
            strategy_summary["passed_by_gas"] = {
                gas: sum(row["results"].get(key, {}).get("reference_comparison", {}).get("passed_by_gas", {}).get(gas, False)
                         for row in records) for gas in ("CO2", "N2", "O2")
            }
        summary["strategies"][key] = strategy_summary
    return {"summary": summary, "records": records}


def write_summary_and_plot(report: dict, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = ["raw_peak_max", "raw_peak_area", "arPLS_peak_max", "arPLS_peak_area"]
    labels = ["Raw + height", "Raw + area", "arPLS + height", "arPLS + area"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    colors = ["#2563eb", "#0f766e", "#9333ea", "#d97706"]
    for index, (strategy, label, color) in enumerate(zip(strategies, labels, colors)):
        values = [row["results"][strategy]["percentages"]["CO2"] for row in report["co2"]["records"]
                  if strategy in row["results"]]
        offsets = np.linspace(-0.14, 0.14, len(values))
        axes[0].scatter(index + offsets, values, s=18, alpha=0.7, color=color)
        axes[0].plot([index - .24, index + .24], [np.mean(values)] * 2, color=color, linewidth=3)
    axes[0].axhspan(99, 100.1, color="#16a34a", alpha=0.1)
    axes[0].axhline(99, color="#16a34a", linestyle="--", linewidth=1, label="99% acceptance boundary")
    axes[0].set(xticks=range(4), xticklabels=labels, ylabel="CO2 weighted signal (%)",
                title="User-identified pure CO2: 54 files", ylim=(95, 100.3))
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    for gas, color in (("N2", "#2563eb"), ("O2", "#d97706")):
        means = [report["air"]["summary"]["strategies"][strategy][gas]["mean"] for strategy in strategies]
        deviations = [report["air"]["summary"]["strategies"][strategy][gas]["std"] for strategy in strategies]
        axes[1].errorbar(range(4), means, yerr=deviations, label=gas, marker="o", color=color, capsize=4)
    axes[1].set(xticks=range(4), xticklabels=labels, ylabel="Weighted signal (%)",
                title="Air: 5 files; mean and population SD", ylim=(0, 100))
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Fixed default coefficients = 1; no response calibration", fontsize=13)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)

    lines = [
        "# 真实样品固定参数对照", "",
        f"生成时间：{report['generated_at_utc']}。", "",
        "54 份用户确认的纯 CO₂ SIF，空气目录直接 4 份 ASC 加子目录“第二个块”1 份，共 5 份。原始文件只读，没有复制或改写。逐文件 SHA-256、处理参数和结果见同名 JSON/CSV。", "",
        "用户确认激光波长均为 532 nm。CO₂ SIF 使用文件内真实校准；空气 ASC 使用文件已有横坐标，未进行二次拟合。54 份 CO₂ 与 5 份空气均完整覆盖 N₂/O₂/CO₂ 窗口。", "",
        "所有策略固定使用默认三种定量气体 N₂、O₂、CO₂，修正系数均为 1；峰窗口分别为 2330±20、1555±25、1388±15 cm⁻¹。arPLS 使用 λ=100000、50 次上限、容差 1e-6。每个窗口均按算法既有规则减去两端连线背景。未按样品真值调整参数。", "",
        "这些输出是加权信号占比。纯样品对照按 CO₂=100%、N₂=0%、O₂=0%、每种气体绝对误差≤1个百分点评估；不意味着完成了混合气体响应系数标定。空气只检查重复性和峰位，不宣称浓度准确度通过。", "",
        "## 纯 CO₂ 对照", "",
        "|策略|CO₂ 平均值|CO₂ 范围|CO₂ 达标|N₂ 零值达标|O₂ 零值达标|三种气体均达标|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in strategies:
        summary = report["co2"]["summary"]["strategies"][strategy]
        co2 = summary["CO2"]
        passed = summary["passed_by_gas"]
        lines.append(f"|{strategy}|{co2['mean']:.4f}%|{co2['min']:.4f}–{co2['max']:.4f}%|{passed['CO2']}/54|{passed['N2']}/54|{passed['O2']}/54|{summary['all_gases_within_1_percentage_point']}/54|")
    lines += ["", "四种默认策略均未使全部 54 份纯 CO₂ 达到 1 个百分点要求。该结果如实保留，不以平均值通过代替逐样品达标。", "",
              "## 空气重复性", "", "|策略|N₂ 平均±标准差|O₂ 平均±标准差|CO₂ 平均±标准差|", "|---|---:|---:|---:|"]
    for strategy in strategies:
        summary = report["air"]["summary"]["strategies"][strategy]
        values = [f"{summary[gas]['mean']:.4f}±{summary[gas]['std']:.4f}%" for gas in ("N2", "O2", "CO2")]
        lines.append(f"|{strategy}|{'|'.join(values)}|")
    lines += ["", "表中标准差为这 5 份文件的总体标准差，不是仪器误差或置信区间。", "",
              "## 窗口峰位", "", "|样品组|气体|raw_peak_max 峰位范围 (cm⁻¹)|", "|---|---|---:|"]
    for group, gases in (("co2", ("CO2",)), ("air", ("N2", "O2"))):
        for gas in gases:
            center = report[group]["summary"]["strategies"]["raw_peak_max"][gas]["peak_center_cm1"]
            lines.append(f"|{group}|{gas}|{center['min']:.5f}–{center['max']:.5f}|")
    lines += ["", "## 适用范围", "",
              "本次验证可以支持导入、文件内校准解析、峰位及固定算法行为的检查。纯度身份和 1% 容差来自用户声明，未提供标准气证书、批次不确定度、环境温压及混合气标定系列。所有修正系数为 1，空气信号比例不能直接当作空气体积分数；纯样品的非目标气体窗口也可能给出正噪声信号。", ""]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--co2", type=Path, required=True)
    parser.add_argument("--air", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {"baseline": SETTINGS, "gas_library": DEFAULT_GAS_LIBRARY,
                     "calibration_override": None, "fitted_parameters": None,
                     "laser_wavelength_nm_user_confirmed": 532.0},
        "interpretation": "Weighted signal fractions, not validated physical concentration; comparison does not confer response calibration",
        "co2": compare(args.co2, "user_identified_pure_CO2"),
        "air": compare(args.air, "user_identified_air"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "file", "sha256", "strategy", "CO2_signal_percent", "N2_signal_percent", "O2_signal_percent",
                         "CO2_peak_cm1", "N2_peak_cm1", "O2_peak_cm1", "CO2_error_pp", "CO2_within_1pp", "N2_within_1pp", "O2_within_1pp", "all_gases_within_1pp"])
        for group in ("co2", "air"):
            for row in report[group]["records"]:
                for strategy, result in row["results"].items():
                    writer.writerow([row["group"], row["relative_path"], row["sha256"], strategy,
                                     *[result["percentages"][gas] for gas in ("CO2", "N2", "O2")],
                                     *[result["peaks"][gas]["center"] for gas in ("CO2", "N2", "O2")],
                                     result.get("absolute_error_percentage_points", ""),
                                     *[result.get("reference_comparison", {}).get("passed_by_gas", {}).get(gas, "") for gas in ("CO2", "N2", "O2")],
                                     result.get("reference_comparison", {}).get("all_gases_within_1_percentage_point", "")])
    write_summary_and_plot(report, args.output)
    print(json.dumps({key: report[key]["summary"] for key in ("co2", "air")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
