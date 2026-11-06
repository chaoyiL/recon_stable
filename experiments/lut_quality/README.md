# LUT 单值兼容性评估

本程序读取已有法向标定样本和 LUT，按原始有符号线性 RGB 色差分配最近的 LUT 节点，在每个占用节点求覆盖原始坡度样本权重最大的圆。它报告不同坡度容差下，任何单值输出都无法覆盖的样本比例。计算不修改原样本或 LUT，也不重新训练模型。

在仓库根目录运行：

```bash
uv run python -m experiments.lut_quality --dry-run
uv run python -m experiments.lut_quality
```

默认读取 `config.yaml` 中的法向标定样本和同目录的 `normal_lut_train.npz`，结果默认保存在配置的 `normal_calibration.work_dir/quality/`（当前为 `assets/normal_calibration/quality/`）。`--dry-run` 只检查输入和估算工作量，不创建结果目录。已有非空结果目录时程序拒绝覆盖；可用 `--out` 指定新目录。

训练 LUT 的 `training_images` 用作参考集，`validation_images` 用作独立留出集。仅在参考集确定每格最优圆心；留出样本使用这些圆心，未见格单独统计。用 `--lut` 选择全量 LUT 时，全部样本均用于参考集，不报告独立留出结果。

```bash
# 坡度容差使用无量纲欧氏距离；姿态等权使每张图总权重相同。
uv run python -m experiments.lut_quality --weighting pose_equal \
  --out assets/normal_calibration/quality_pose_equal

# 选择完整 LUT，评估全样本的单值兼容性。
uv run python -m experiments.lut_quality --lut assets/normal_calibration/normal_lut.npz \
  --out assets/normal_calibration/quality_all
```

默认容差为 `0, 0.025, 0.05, 0.1, 0.2`，可通过 `--taus` 修改；`--detail-tau` 指定逐格展示所用容差，必须包含在 `--taus` 中。`--split all` 显式将全部样本作为参考集。其他选项见 `uv run python -m experiments.lut_quality --help`；同步项目环境后也可使用 `uv run lut-quality`。

结果包括 `quality.json`、`curves.csv`、`per_bin.npz`、`top_bins.json`、`manifest.json` 和对应曲线、逐格、切片、样本数及法向覆盖图片。逐格结果仅包括参考集占用的节点，保留所有原始样本和重复样本权重；图表中样例数量不会影响指标计算。

该指标衡量在所选 RGB 节点分箱和样本分布下，单个坡度答案能解释多少原始记录。它不是 LUT 的插值预测误差、法向角误差，也不替代标定的发布验证；比较不同数据集时应同时考虑采样覆盖与姿态分布。方法依据及数值细节见 [LUT 质量评估方法与复现交接](../../docs/LUT质量评估方法与复现交接.md)。
