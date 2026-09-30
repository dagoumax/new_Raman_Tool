# 一致性、参考样品与发布验证

## 验证目标

相同光谱、相同导入选项、相同基线参数、相同气体库和定量窗口，
应在 Qt 单文件、Qt 批量、命令行和终端得到一致的结果。
用户已明确浓度允许存在偏差，当前优先保证可重复性与入口一致性。
此前声明的纯 CO₂ 1 个百分点容差仍保留在报告中作对照，不再作为软件完成的必要条件。

算法输出为所选气体的归一化加权信号占比。峰高与面积均减去窗口两端连线背景。
全为 1 的默认修正系数未经响应标定；不同策略可以给出不同数值。
不允许通过只启用单一气体，使必然为 100% 的归一化结果充当纯度验证。

## 回归与打包

~~~powershell
uv sync --locked --extra qt --extra dev
uv run pytest
uv run raman-tool validate-standard tests/fixtures/standards/synthetic-mixture.json --report output/synthetic-report.json
uv build
~~~

参考报告默认拒绝覆盖，确需替换可添加 --overwrite。对照不通过的退出码为 1。
源码发行包包含测试及合成标准；wheel 包含程序，不附带本机实验数据。
自动测试配置覆盖 Windows/Linux、Python 3.10/3.12、最低依赖版本和离屏 Qt。
远端 CI 需要推送后运行；本机结果不代表远端矩阵已通过。

## 参考清单

以 tests/fixtures/standards/synthetic-mixture.json 为模板：

- sample_kind：synthetic 或 measured，必须明确声明。
- sample_id、reference_source、acquisition_conditions：编号、已知真值来源和采集条件。
- file：相对清单的光谱路径或绝对路径。
- gas_library：固定气体库快照，至少两个启用定量的气体，包含峰中心、半窗口及修正系数。
- expected_percentages：覆盖所有定量气体，总和为 100。
- tolerance_percentage_points：统一正数或逐气体的绝对百分点容差。
- strategy：peak_max 或 peak_area。
- import_options：可选 row_groups、col_merge、row_mode、calibration。
- baseline_options：可选 method、lam、degree、max_iter、tol。
- coefficient_provenance：实测清单必填，含 status（uncalibrated/calibrated）和 source；
  标定系数还需 applicability，说明适用条件。
- expected_peak_centers：可选逐气体的 center 与 tolerance_cm1，检查独立峰位约束。

报告记录源文件/清单 SHA-256、参数、真值、计算值、偏差和比较结果。
缺失窗口、非有限数据、无有效信号或无效参数会明确报错，不会静默替换成零。
合成真值通过证明软件数值行为；实测对照不等同于完成仪器准确度或混合气体响应标定。

## 用户样品

纯 CO₂ 目录：54 份 SIF；空气 gc/tantou/12yz：直接 4 份 ASC，
子目录“第二个块”1 份，共 5 份。用户确认激光波长均为 532 nm。
SIF 使用文件内校准；ASC 使用已有横坐标，不按预期峰位回拟合。

固定参数对照采用默认 CO₂/N₂/O₂ 三个定量窗口，修正系数均为 1。
分别计算原始数据及 arPLS（lam=100000，max_iter=50，tol=1e-6），
各自运行峰高和面积策略。纯 CO₂ 对照目标为 100/0/0%，每种气体 1 个百分点；
空气未给定独立精确组成，只比较峰位、加权占比和重复性。

~~~powershell
uv run python validation/compare_reference_strategies.py --co2 "<纯CO2目录>" --air "<空气目录>" --output validation/local-reports/comparison.json
~~~

逐文件报告、图表及本机路径由 .gitignore 排除。原始样品不复制、不改写。
所有默认策略均有未达到此前 1% 对照容差的纯 CO₂ 文件；偏差保留在报告中。
参考图表中的标准差为这组文件的总体标准差，不是仪器误差或置信区间。

## 会话与线程

默认最多 32 个撤销状态、128 MiB 数组；始终保留原始数据与最近状态。
旧撤销状态裁剪会显示提示；审计日志完整保存在会话临时流，界面仅显示最近 200 条。
缓存默认 16 个驻留会话、256 MiB，溢出会话暂存到磁盘（默认 1 GiB）。
磁盘满时保留已有结果并阻止继续增长，可导出后清理文件列表。
这些是当前运行的临时资源，退出前应导出结果。

自动/手动基线校正在后台执行；快速切换文件忽略过期结果；批处理启动时固定气体库。
取消在读取和数值计算之间生效，正在执行的单次数值求解结束后才退出。
窗口关闭会继续处理事件，待后台任务结束后关闭。

## 代码位置

- workflows.py 与 validation.py：共用导入流程和校验。
- readers/image_reader.py、readers/text_reader.py：共用图像/文本处理。
- qt_workers.py：后台任务和生命周期；qt_dialogs.py：独立对话框。
- history.py：快照、审计、有限缓存与磁盘暂存。
- reference_validation.py：参考清单与可追溯报告。
