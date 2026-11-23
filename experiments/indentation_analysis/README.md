# 球面拟合与法向夹角分析

输入 `recon`、`recon-batch` 或 `local-recon` 输出的局部重建 NPZ，按已知球半径选择球面区域、拟合球心，再计算球心到整体基准面的投影连线夹角。

```bash
uv run python -m experiments.indentation_analysis \
  assets/reconstruction/pc/3b2_batch/mlp6.npz
```

默认从项目 `config.yaml` 读取真实球半径，按候选接触区质心的网格行、列顺序逐个对应，单位为 mm：

```yaml
indentation_analysis:
  sphere_radii_mm: [2.0, 1.5, 1.0]  # 当前上、中、下三个球
  depth_scan:
    step_mm: 0.02
    min_fit_points: 100
    stable_steps: 5
    rmse_atol_mm: 0.005
    rmse_rtol: 0.01
```

半径列表长度须与 `--count` 一致，各值为有限正数。默认在输入文件旁生成 `<文件名>_analysis/`。通过 `--config` 指定配置，或用 `--radii-mm` 临时指定半径：

```bash
uv run python -m experiments.indentation_analysis /path/to/result.npz \
  --output-dir /path/to/analysis --count 3 --radii-mm 2 1.5 1 \
  --threshold-mm 0.2 --min-region-points 100
```

单球使用 `--count 1 --radii-mm 3` 指定半径 3 mm。单球图中存在额外独立的小变形区时，可增加 `--largest-contact`，按几何点数选择最大接触区；报告保留其余区域的点数和质心。默认按指定区域数量严格匹配。

输入字段为 `xyz_out`、`displacement`、`displacement_vectors`、`valid_mask`、`photometric_valid_mask`，均在同一规则二维网格上。XYZ 与位移单位为 mm；负位移表示压入，整体基准面由 `xyz_out - displacement_vectors` 恢复。

候选区域使用 `valid_mask & (displacement < -threshold_mm)` 的八邻域连通分量，按各区网格行列的算术均值排序编号。默认位移阈值为 0.2 mm，每区至少有 `min_region_points` 个几何点（默认 100）。各区的光度有效点构成扫描候选集。

每个区域从最深候选点开始，以 `step_mm` 向浅处扩展点集。每层在全部当前深度点上固定真实半径 R，等权最小化径向残差 `||P-C||-R`，独立求解球心 C。拟合最少点数由 `depth_scan.min_fit_points` 配置。

连续 `stable_steps` 次扩展形成 `stable_steps + 1` 帧窗口。要求每帧拟合成功、每步新增点、深度步长完整，且窗口满足：

$$
\max(\mathrm{RMSE})-\min(\mathrm{RMSE})
\le \max(\texttt{rmse\_atol\_mm},\ \operatorname{mean}(\mathrm{RMSE})\,\texttt{rmse\_rtol}).
$$

项目配置使用连续 5 次扩展、6 帧窗口，绝对容差 0.005 mm、相对容差 1%。当前绝对容差覆盖 `1b` 的 25 组图像/LUT 扫描。首次满足条件时，采用窗口末帧的球心、深度点集、RMSE 和 `MSE=RMSE²`，MSE 单位为 mm²。拟合失败、点集重复或末尾短步会中断连续计数。

扫描到全候选范围仍未收敛时，状态为 `depth_not_converged`，最终球心、RMSE 和 MSE 为 `null`，完整扫描曲线继续保存。阈值保持配置值，由使用者调整后重算。配置中的深度参数随分析报告保存。

球心通过整体基准面网格的三次样条做最近点投影。投影使用完整 4×4 有效邻域的局部插值，连线方向由投影点指向球心，计算 0–180° 的有向夹角。报告保留球心相对基准面的有向距离，对负侧或接近方向翻转的情况给出提示。三个或以上有效球心同时计算方向共面性、各连线端点的最佳拟合平面，以及投影到该平面后的夹角。

输出目录包含：

- `analysis.json`：候选区质心、取点方法、拟合质量、球心、方向检查、投影和夹角。
- `sphere_fits.csv`：每区一行，记录候选/拟合点数、收敛深度、RMSE、MSE、球心、条件数及状态。
- `depth_scan.csv`、`depth_scan.json`：扫描到收敛或终点的全部层，记录点数、RMSE、MSE、球心和拟合状态。
- `fit_selection.npz`：各接触区的候选点 mask 和实际拟合点 mask，与输入 NPZ 的网格逐点对应。
- `analysis.png`：深度扫描曲线与收敛标记、候选轮廓与实际取点、投影连线的 X-Z 图。
- `fit_check.png`：灰色候选点、彩色实际拟合点、黑色拟合球面与紫色投影线的三维叠加图。

重复运行更新这些产物，并清理此前的 `depth_sweep.csv`。退出码 0 表示各区拟合和投影完成；1 表示部分拟合或投影失败，此时保留可检查的取点与已有拟合结果，对应夹角为 `null`；2 表示输入或候选接触区识别错误。

球半径是输入先验，拟合变量为球心。收敛值描述最早稳定深度范围的球面贴合程度；阈值和最少点数会影响取值深度。浅平面也可能出现稳定低误差，结果应结合扫描深度、支持点数及三维拟合图解释。球心和夹角的绝对准确度需结合重建误差及独立测量评估。

## RMSE 与曲率联合深度扫描

`curvature_depth_sweep.py` 是联合实验入口。每次运行从重建点云及各方法已有的 `*_analysis/analysis.json`、`fit_selection.npz` 读取真实半径和候选区域，依次计算 RMSE 深度扫描、局部曲率、深度统计和相邻球心距离，最后生成一张联合总图。

单球 `1b_batch` 的球半径为 3 mm，当前图展示到 d=1.75 mm：

```bash
uv run python -m experiments.indentation_analysis.curvature_depth_sweep \
  assets/reconstruction/pc/1b_batch --upper-plot-depth-mm 1.75
```

三个球冠及五种 LUT 的实验使用相同入口：

```bash
uv run python -m experiments.indentation_analysis.curvature_depth_sweep \
  assets/reconstruction/pc/3b2_batch \
  --step-mm 0.02 --neighborhood-radius-mm 0.5 --depth-band-mm 0.1
```

参数与默认值：

| 参数 | 默认值与含义 |
|---|---|
| `--config` | 项目 config.yaml，读取扫描与收敛参数 |
| `--step-mm` | 覆盖配置中的扫描步长 |
| `--min-fit-points` | 覆盖完整曲线每层拟合的点数下限；收敛仍遵守配置中的最低点数 |
| `--neighborhood-radius-mm` | 0.5 mm，曲率拟合三维邻域半径 |
| `--min-neighbors` | 12，局部曲率拟合最少观测点数 |
| `--depth-band-mm` | 0.1 mm，局部深度统计带宽 |
| `--min-band-points` | 8，曲率统计最少有效点数 |
| `--upper-plot-depth-mm` | 覆盖第一个球冠的展示上限；R=2 mm 默认展示到 0.8 mm，其余尺寸自适应完整范围 |
| `--output-dir` | 默认写入 batch 的 `analysis/`；可指定新的输出目录 |

### 深度扫描与固定半径拟合

令光度有效候选点的正压入量为 `D_i=-displacement_i`，最深压入量为 `D_max=max(D_i)`。扫描自变量 d 是从最深处向浅处扩展的范围，单位为 mm。每步点集为：

$$
M(d)=M_{candidate}\cap\{D_i\ge D_{max}-d\}.
$$

扫描包含 d=0、指定步长和全候选范围的精确终点。深度沿用各点相对基准曲面的法向位移约定，使用完整候选区，可观察逐步纳入外圈过渡点时的拟合变化；按同一收敛判据标出首个平台。

每步在当前点集上固定真实半径、等权最小化径向残差，球心朝向由该步点集的平均基准法向确定。点数不足时记录 `insufficient_points`。各步同时保存球心 XYZ、相对收敛球心的偏移和 Jacobian 条件数。联合实验保留完整曲线；常规分析在首次收敛处停止。

### 曲率与深度统计

曲率直接由重建 XYZ 的局部形状估计。以每个候选点为中心，搜集指定三维欧氏半径内的光度有效观测点，用 `exp(-2(distance/radius)^2)` 加权做 PCA，得到与基准正法向一致的切平面，再拟合六参数二次图面。拟合要求支持点达到下限、矩阵满秩且条件数至多为 `1e6`。空间邻域覆盖完整观测曲面，深度带用于统计拟合中心。

从二次图面的完整第一、第二基本形式求出升序主曲率 `k_min`、`k_max`，单位为 mm⁻¹，并逐点计算 `K=k_min*k_max`，单位为 mm⁻²。项目采用朝向球心的球冠法向为正，理想球冠满足：

$$
k_{min}R=k_{max}R=1,\qquad KR^2=1.
$$

图中展示归一化高斯曲率：`K<0` 表示两主曲率异号的局部鞍形。虚线水平参考值为理想球面的 1，实线零值用于观察符号变化。主曲率及其统计值继续保存在数值产物中。

每个 d 统计中心位于 `[d-bandwidth/2,d+bandwidth/2]` 的局部深度带，按重建表面真实面积加权取中位数；高斯曲率与两条主曲率分别统计。CSV 同时保留 `offset<=d` 的累计统计，其中 `offset=displacement-min(displacement)`，与 RMSE 使用相同的深度选区。有效点数不足时曲率指标记录为 `null`，保留支持点数和面积。

### 相邻球距离

相邻关系沿用接触区的网格顺序，三个球对应 1–2、2–3 两组。对同一 LUT、同一扫描深度 d 的两个拟合球心，计算三维欧氏距离 `L_i(d)=||C_(i+1)(d)-C_i(d)||₂`，单位为 mm，图中展示标量距离趋势。各球的 d 均从自身最深候选点起算，表示各自向外扩展相同深度范围。

距离只使用两球共有的精确深度层，各自独有的全区终点留在球面扫描结果中。任一球在某层拟合失败时，该层距离记为 `null`，曲线形成断点。距离子图采用两球展示上限中的较小值，完整距离序列保存在 `depth_sweep.json` 的 `adjacent_sphere_distances`。

距离输出按同一 LUT 内的球数启用。单球模式保留 RMSE、高斯曲率和累计点数三行，并跳过距离子图及 JSON 中的距离定义、距离序列字段。

### 产物与结果解释

一次运行在输出目录中生成：

- `rmse_curvature_vs_depth.png`：从上到下依次展示 RMSE、归一化高斯曲率、相邻球心距离和累计点数。距离行按相邻球对分栏，其余各行按球冠分栏；单球使用 RMSE、高斯曲率和累计点数三行。
- `depth_sweep.csv`、`depth_sweep.json`：本次重新计算的固定半径深度扫描，包含收敛参考、每层点数、球心、误差和状态；多球模式的 JSON 另存相邻球距离定义与序列。
- `curvature_depth_sweep.csv`、`curvature_depth_sweep.json`：本次 RMSE、局部带与累计曲率、面积、分位数、负高斯曲率面积比例及方法定义。
- `curvature_maps.npz`：各 LUT 在原网格上的主曲率、有效 mask、邻居数量、局部拟合误差、条件数和面积；有效 mask 之外的曲率为 NaN。

展示上限仅用于总图，CSV 和 JSON 保留完整扫描。每次从原始重建和候选区域重新生成全部报告；指定新输出目录同样可直接运行。

曲率来自有限空间尺度下的二次近似，邻域半径影响平滑程度和估计偏差；深度带宽控制统计中心分组。图中的转折适合按区间解释，空间尺度与深度带宽分别记录。各 LUT 沿用各自最深候选点归零，跨方法比较需结合 JSON 记录的最深压入量。

## 球冠拟合三维检查

`sphere_fit_check.py` 是独立绘图入口，读取 batch 中各方法的重建 NPZ、`*_analysis/analysis.json` 和 `fit_selection.npz`，按已保存的球冠选区、球心、真实半径与基准面投影生成一张拟合总图：

```bash
uv run python -m experiments.indentation_analysis.sphere_fit_check \
  assets/reconstruction/pc/1b_batch
```

默认输出到 `batch_dir/analysis/sphere_fits.png`，通过 `--output /path/to/sphere_fits.png` 指定图片路径。每次更新这张三维拟合图，重建数据和深度分析报告继续保留。

各面板共用相机 XYZ 范围、视角和径向残差色标。彩色点为收敛深度内的实际拟合点，颜色表示 `||P-C||-R`；黑色网格为拟合球面，黑叉为球心，紫色线为球心至基准面的投影线。灰色显示当前空间窗口内被剔除的候选点，标题中的点数为拟合点数/全部候选点数。未收敛的面板明确标注状态。深度扫描入口中的展示上限仅控制深度曲线。

每个拟合面板还标出球心有符号高度 `h_c` 和由球心推导的压入深度 `d`，单位均为 mm。令 C 为球心、Q 为 C 在当前整体基准曲面上的最近法向投影，基准正法向 n₀ 确定侧别，则：

$$
h_c=\operatorname{sign}((C-Q)\cdot n_0)\,\|C-Q\|,
\qquad d=R-h_c.
$$

球心位于基准面正侧时 `h_c>0`，位于负侧时 `h_c<0`；负侧球心对应的 d 大于 R。计算采用完整三维投影距离。基准曲面由 `xyz_out-displacement_vectors` 恢复，法向朝向沿用原选区的基准正法向。投影缺失时，图中以 `d=N/A` 保留该状态。

## 单球图像集批量重建与 RMSE 分布

`sphere_batch_rmse.py` 逐图使用五种 LUT 重建、选取最大主接触连通区、拟合已知半径球面，并汇总每张图的径向 RMSE。启动时复用 `recon-batch` 的版本选择规则，冻结五组模型和材料网格路径，保持整组实验使用相同标定版本。

```bash
uv run python -m experiments.indentation_analysis.sphere_batch_rmse \
  assets/test_pics/1b
```

输入默认匹配目录中的 `*.npz`，可用 `--pattern` 指定其他图像格式。球半径默认读取 `normal_calibration.sphere_radius_mm`，当前为 3 mm；`--radius-mm` 可临时指定已知半径。主接触区的负位移阈值默认 0.2 mm，候选区几何点数至少 100，可用 `--threshold-mm`、`--min-region-points` 调整。拟合和收敛参数统一读取 `indentation_analysis.depth_scan`。`--config`、`--reference-image` 沿用重建配置约定。

默认产物根目录为 `reconstruction.output_dir/pc/<输入目录名>_series_batch/`，`--output-dir` 可覆盖它。当前输入对应 `assets/reconstruction/pc/1b_series_batch/`；每张图片的点云、预览和拟合报告都集中在这个实验根目录内：

```text
1b_series_batch/
  rmse_per_image.csv
  rmse_statistics.csv
  rmse_summary.json
  rmse_distribution.png
  1b1/
    table.npz, mlp3.npz, mlp5.npz, mlp6.npz, mlp8.npz
    pics/                 # 五种方法的重建预览
    table_analysis/       # 分析报告、深度扫描 CSV/JSON 与拟合掩膜
    mlp3_analysis/
    ...
  1b2/
  ...
```

`rmse_per_image.csv` 每个图像/方法组合一行，记录重建和拟合状态、误差阶段、候选/拟合点数、收敛深度、`converged_rmse_mm`、`converged_mse_mm2`、球心和压入深度。单次失败会写入明细，后续组合继续执行。每张图片结束后更新明细与 JSON。

主指标使用首次稳定窗口末帧的全部深度拟合点：`RMSE=sqrt(mean((||P-C||-R)^2))`，`MSE=RMSE²`。阈值保持配置值；未收敛状态为 `depth_not_converged`，对应主指标为 `null`，CLI 列出案例供调整阈值。投影失败而深度扫描收敛时，RMSE/MSE 进入统计，投影深度留空。

`rmse_statistics.csv` 每个方法、每种指标一行，包含有效图像数 n、总图像数、均值、总体方差与标准差（`ddof=0`）、样本方差（`ddof=1`）、中位数、极值、四分位数、5%/95% 分位数和 IQR。每张图等权，并提供按拟合点数加权的 pooled 指标。`unit` 和 `variance_unit` 区分 RMSE 的 mm/mm² 与 MSE 的 mm²/mm⁴。

`rmse_distribution.png` 展示收敛 RMSE 和 MSE 的分布，按样本编号配色。`depth_scan_overview.png` 按样本行、LUT 方法列展示所有扫描曲线和首个收敛点，未收敛案例保留完整曲线并标注。`rmse_summary.json` 保存参数、来源与哈希、明细及统计。

已有点云可直接重算分析：

```bash
uv run python -m experiments.indentation_analysis.sphere_batch_rmse \
  assets/test_pics/1b --analysis-only
```

默认所有产物仍集中于 `assets/reconstruction/pc/1b_series_batch/`。退出码 0 表示所有组合完成，1 表示存在未收敛或其他案例失败，2 表示输入或配置错误。
