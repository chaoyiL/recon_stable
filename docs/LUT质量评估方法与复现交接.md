# LUT 质量评估方法与复现交接

整理日期：2026-11-03。依据当前 `experiment/lut_quality` 实现整理；指标版本为 `lut_single_value_incompatibility_v1`。本次在 `59b078e` 基础上裁剪输出，报告与 manifest 的 `format_version` 为 2，单值不可兼容率的数值定义保持不变。

这份说明供另一个 agent 复现现有评估。输入是已经归档的原始标定样本与 LUT，输出是不同坡度容差下的失败比例、采样覆盖和独立留出结果。运行时复用这些数据，不重新渲染、不重新拟合 LUT，也不改变原标定的通过或失败状态。

## 1. 要复现的指标

该实验检验全局的“有符号线性 RGB 残差 → 几何坡度 `(p,q)`”映射，位置和压入姿态不作为 LUT 输入。只计算 `incompatibility`，下文记为 `A`，以及独立姿态上已有圆心的转移结果。

它回答：**同一个 RGB 格子里的原始标定记录，如果只能给出一个坡度答案，最好情况下仍有多大比例无法满足容差？** 比如一个格子里最多有 80% 的记录能被同一个答案解释，该格的 `A` 就是 20%。比例越低越好。

当前流程是“原始样本 → 最近 RGB 节点分组 → 统计每格全部原始坡度 → 求最佳覆盖圆”。**它没有以每个 `dI` 为中心，在完整 LUT 中搜索某个颜色距离以内的节点。** `tau` 控制坡度容差；当前没有独立的 `dI` 邻域半径。

LUT 提供分箱用的颜色范围、节点数和保存的姿态划分。原始记录保留了建表平均之前一对多的对应关系，填好的 LUT 本身不保存这些分布。`A` 描述当前样本与分箱，不衡量实际查表预测的准确率，也不是连续插值映射的误差下界。

本指标不是法向角误差、RGB 近邻对计数或已删除的 NormDiff。默认容差也不是标定合格阈值，本实验不承担正式 LUT 发布判定。

## 2. 源码与交接数据

### 源码对应关系

| 文件 | 复现时查看的内容 |
|---|---|
| [metric.py](../experiment/lut_quality/metric.py) | `bin_samples`、`maximum_disc`、`evaluate_quality`、`evaluate_holdout`、`split_samples`，唯一的指标数值实现 |
| [runner.py](../experiment/lut_quality/runner.py) | 输入检查、CLI、实际路径、报告字段与运行状态 |
| [plots.py](../experiment/lut_quality/plots.py) | 图表的切片、覆盖与样例展示规则 |
| [experiment/config.py](../experiment/config.py) | `LUT_QUALITY_DEFAULTS`、`resolve_lut_quality_settings`，参数校验和覆盖规则 |
| [normal_mapping.py](../utils/normal_mapping.py) | `NormalCalibration` 文件格式与法向约定；本实验不调用实际插值查询 |
| [lut_results.py](../utils/lut_results.py) | 标定目录、物理快照、后端及运行签名检查 |
| [calibration_samples.py](../utils/calibration_samples.py) | 原始材料网格样本如何产生，仅用于理解输入语义 |
| [实验 README](../experiment/lut_quality/README.md)、[实验配置](../configs/lut_quality.json) | 常用操作与默认参数 |

### 应交给复现 agent 的文件

交接一个明确的标定运行目录 `RUN`，至少包含：

```text
RUN/
  manifest.json
  normal_samples.npz
  normal_lut_train.npz
  normal_lut.npz                 # 若要比较正式全量 LUT，一并交接
  normal_lut_diagnostic.npz      # 若要比较失败后的全量诊断 LUT，一并交接
  evaluation/lut_quality*/      # 若要逐项核对旧报告，一并交接
```

`manifest.json` 应保留原始 `config` 快照、`run_signature` 和状态。旧诊断目录只有 `config.json` 时，入口也能将它作为未发布结果读取；默认仍要求样本和训练 LUT 完整。即使使用 `--samples` 或 `--lut` 覆盖路径，`load_lut_result` 仍先检查 `RUN/normal_samples.npz` 与 `RUN/normal_lut_train.npz`。

当前后端为 `analytic`、`xense`，旧 `mujoco` 名称归一化为 `xense`。结果快照包含 `surface` 时，现有加载器要求它是平面配置；这不是曲面迁移实验的质量评估入口。

`results*/` 与 `tests/` 受仓库忽略规则影响，新的 Git 克隆不一定有这些数据和测试。交接时要另外传递所选运行目录；不要让接收方用当前仿真配置重新生成样本来代替原始输入。

## 3. 输入语义与读取规则

### 原始样本

`normal_samples.npz` 中下列四项是必要字段：

| 字段 | 形状 | 含义 |
|---|---|---|
| `color_residual_samples` | `[N,3]` | 按 RGB 排列的有符号线性颜色残差 |
| `slope_samples` | `[N,2]` | 与每条残差配对的几何坡度 `(p,q)` |
| `image_index` | `[N]`，整数 | 每条样本所属完整压入姿态在 `image_paths` 中的索引 |
| `image_paths` | `[P]`，唯一字符串 | 姿态键；用于匹配 LUT 保存的训练/验证名单，不需要打开对应图片 |

`N` 必须大于零，颜色和坡度必须全部有限，索引必须有效。所有原始行都参与所属分组的评估，不根据 LUT 的 `original_valid`、方差、置信度或拟合时的 MAD 规则再次筛选。相同坡度的重复行也保留其全部权重。

现行材料网格标定中，参考图与压入图先经过 `sRGB8 → inverse_sRGB`，再分别在同一材料点采样并相减：

$$
dI_i=I_{\mathrm{observed,linear}}(u_i)-I_{\mathrm{reference,linear}}(u_i).
$$

这是归档样本的来源说明。质量检验直接读取保存的 `dI`，不要再做 gamma 解码、绝对值、0–255 缩放或减背景。样本生成时已经执行接触、可见性、饱和等有效性筛选，本实验不重新执行这些筛选。

坡度无量纲；对应几何法向为

$$
n(p,q)=\frac{(-p,-q,1)}{\sqrt{1+p^2+q^2}}.
$$

`tau` 是 `(p,q)` 的欧氏距离，不是角度。同一 `tau` 在不同倾角处不对应固定法向角误差。

### LUT 与来源检查

LUT 用 `NormalCalibration.load` 读取，支持文件 `format_version=4/5`，必须声明 `color_residual_mode='signed'`。主要字段是 `[S,S,S,2]` 的 `slopes`、三个颜色通道的 `color_min/color_max`、`variances`、`sigma_ref2` 和 JSON `metadata`。LUT 数值读取后按现有实现转成 `float32`；方差和置信度不用于本实验加权或剔除样本。

样本读取后颜色与坡度转成 `float64`。样本与 LUT 同时声明了以下信息时必须一致：`run_signature`、`simulation_config_signature`、`light_geometry_model`、`coordinate_chart_id`、`color_space`、`residual_domain`、`image_encoding`、`observation_model_id`、`backend`。显式颜色域必须是 `linear_rgb`；运行 manifest 有签名时，也会与所选样本和 LUT 对照。缺少签名会记录来源无法完整核验的警告，不靠代码哈希或依赖版本一概拒绝运行。

## 4. 先划分完整姿态，再确定权重

默认 `split=auto`：

1. 若所选 LUT 的 `metadata.validation_status == 'training_only'`，用其保存的 `training_images` 作为 reference 分组，`validation_images` 作为 holdout 分组。两份名单必须非空、互不相交，并覆盖归档中的全部样本。沿用整个姿态的划分，不重新随机划分点。
2. 正式全量 LUT、全量诊断 LUT，或其他非 `training_only` 状态的 LUT，全部样本作为 reference，报告 `holdout=null`。旧验证名单不会变成全量拟合后的独立留出集。
3. 显式 `split=all` 也使用全部样本，取消内部独立验证。

reference 是用于求最优圆心的分组；默认训练 LUT 下它就是训练组。报告中分别记录 `calibration_status` 和 `lut_status`，失败标定也可以有有效的训练组/留出组结果。

权重在 reference 与 holdout 内分别计算。令当前分组中姿态 `j` 的样本数为 `N_j`：

$$
w_i=\begin{cases}
1,&\texttt{sample},\\
1/N_{\mathrm{pose}(i)},&\texttt{pose\_equal}.
\end{cases}
$$

`pose_equal` 使每个有样本的完整姿态总权重相同，同时改变格内最优圆和格间聚合权重。它不是每格等权、不是面积加权，也不自动平衡法向覆盖。

## 5. RGB 分箱与核心指标

### 5.1 归属最近节点

对每个通道，用所选 LUT 自己的颜色范围计算：

$$
x_i=\operatorname{clip}\!\left(\frac{dI_i-c_{\min}}{c_{\max}-c_{\min}},0,1\right),\qquad
k_i=\operatorname{rint}\bigl((S-1)x_i\bigr).
$$

`np.rint` 使用半偶舍入，不能替换成普通四舍五入。RGB 顺序不变；三个节点坐标展平为

$$
b_i=(k_{R,i}S+k_{G,i})S+k_{B,i}.
$$

分箱使用 `float64` 运算和加载后 LUT 的颜色界限。越界样本夹紧到边缘节点并继续计算，同时单独报告按当前样本权重计算的 `out_of_range_fraction`。分箱不依据样本重新估计颜色范围。

### 5.2 每格寻找最有利的单值输出

令格 `b` 中的原始坡度为 `s_i=(p_i,q_i)`，总权重为 `W_b`。对每个容差 `tau`，求：

$$
C_b(\tau)=\max_{c\in\mathbb R^2}\sum_{i:b_i=b}w_i\,\mathbf1[\|s_i-c\|_2\le\tau],
\qquad
A_b(\tau)=1-\frac{C_b(\tau)}{W_b}.
$$

圆心可以位于任意坡度位置，不限于实际样本。全局结果是按格内样本总权重聚合：

$$
A(\tau)=\frac{\sum_b W_b A_b(\tau)}{\sum_b W_b}.
$$

不能直接对所有占用格子的 `A_b` 做算术平均。

例如同一格中 5 条坡度为 `(-1,0)`、5 条为 `(1,0)`，`tau=0.1`：最优单值只能覆盖一半，故 `A=0.5`。这里统计的是全部原始记录的权重，不是把两种不同坡度各算一次。

### 5.3 最大权重圆的现有算法

`maximum_disc` 用精确的加权角度扫描，不做下采样、聚类近似或候选截断：

1. 用 `np.unique(points, axis=0)` 合并完全相同的坡度位置，把重复行权重相加。
2. `tau=0` 时直接取权重最大的相同坡度组；只有一个不同坡度时同样直接返回。
3. 先尝试所有不同坡度包围盒的中点。若该圆已包含全部位置，立即返回全覆盖结果。
4. 否则依次让每个不同坡度 `a` 位于圆周。任意其他点 `s` 距离为 `d≤2*tau` 时，允许的圆心方向区间为

   $$
   \alpha\pm\arccos\!\left(\frac{d}{2\tau}\right),\qquad
   \alpha=\operatorname{atan2}(s_q-a_q,s_p-a_p).
   $$

   将跨越 `0/2π` 的区间正确计入初始覆盖权重，排序扫描所有进入/离开事件，取覆盖权重最大的位置。
5. 重新按返回圆心计算实际覆盖权重。`evaluate_quality` 还比较实际 LUT 节点圆的覆盖量；若节点覆盖更多，改用节点输出作为最优圆心，处理浮点扫描的数值偏差。

格内不同坡度数为 `U_b` 时，扫描最坏复杂度为 `O(U_b² log U_b)`，需要对所有容差、所有占用格计算。工作量由高占用格主导，减少 `top_bins` 不会减少核心计算量。实现用 CPU/Numba，缺少 Numba 时执行同一 Python 算法，不需要 CUDA。

数值边界与平局规则也属于复现口径：

- `tau>0` 的最终闭圆判定半径为 `tau + 1e-12*max(1,tau)`；`tau=0` 按完全相等，不加容差。扫描几何本身使用 `tau`。
- 扫描将相差不超过 `1e-13` 的角度事件视为同组，先加进入事件、评价覆盖，再减离开事件，以包含相切边界。
- 扫描事件中的最优覆盖更新条件为大于旧值 `1e-14`；零容差并列组取 `np.argmax` 的首项。扫描依次遍历 `np.unique` 的排序结果。
- 多个圆心可能取得相同覆盖量。另写求解器时仅得到相同 `A`，不保证留出集转移结果一致；要逐项重现旧报告，优先直接复用 `maximum_disc` 和节点覆盖补充判断。

### 5.4 与完整 LUT 颜色邻域搜索的区别

当前每条原始样本只归属一个最近 RGB 节点。每格的坡度分布来自所有归属样本；不同样本不会因为颜色略有不同而被划成滑动重叠邻域。未采到原始记录的格子不参与 `A` 聚合。

“给定 `dI`，从完整 LUT 收集某个颜色距离范围内的所有节点坡度，再求最佳覆盖”会使用另一套统计对象，其中还可能包含补值节点。它需要单独定义颜色距离、颜色半径、节点权重、查询中心和聚合方式。当前保留上述原始样本分箱算法，没有实现这套邻域搜索。

## 6. 独立留出集怎样计算

只在 reference 样本上为每个格、每个 `tau` 求圆心。holdout 使用相同分箱和已经选定的圆心，不能拿留出真值重新选圆。

| 留出字段 | 定义 |
|---|---|
| `unseen_bin_fraction` | 留出样本落入 reference 未占用格的加权比例 |
| `transferred_disc_failure_or_unseen` | 已见格但超出 reference 圆，加上所有未见格，除以留出总权重 |
| `transferred_disc_known_bin_error` | 只在已见格中统计超圆比例；没有已见格时为 `null` |

留出转移失败率不是重新优化的 `A`，也不是 LUT 的必然下界。reference 的圆心即使有并列最优，也不能根据留出表现挑选。

## 7. 参数与运行入口

先复用仓库 `.venv`，依赖以 `pyproject.toml`、`uv.lock` 为准。评估需要 NumPy、Matplotlib；Numba 可用时自动加速。仅在依赖缺失时再按锁文件安装，例如 `uv sync --cache-dir .uv-cache --locked --extra calibration`。

当前 [configs/lut_quality.json](../configs/lut_quality.json) 的默认值如下：

```json
{
  "format_version": 1,
  "lut_quality": {
    "taus": [0, 0.025, 0.05, 0.1, 0.2],
    "detail_tau": 0.1,
    "weighting": "sample",
    "split": "auto",
    "top_bins": 8,
    "examples_per_bin": 12
  }
}
```

数值参数优先级为显式 CLI > 实验 JSON > 内置默认值，CLI 未提供的 `None` 不覆盖 JSON。`taus` 至少两个不同的非负有限数，计算前升序排序；`detail_tau` 必须属于其中；两个样例数量必须是正整数。`detail_tau` 和样例数量只控制展示，不改变全量指标。

**当前入口的路径行为：**默认样本、LUT、输出分别取 `RUN/normal_samples.npz`、`RUN/normal_lut_train.npz`、`RUN/evaluation/lut_quality`，覆盖它们要用 `--samples`、`--lut`、`--out`。配置解析器虽然接受 `samples/lut/output_dir` 字符串，当前 `runner.main` 没有将它们用于实际路径选择；复现时不要依赖这些 JSON 路径。CLI 相对路径以当前工作目录为准。本实验不加载当前 `configs/analytic.json` 或 `configs/xense.json` 补值。

从仓库根目录执行，先把 `$lutRun` 换成实际交接的那个运行目录：

```powershell
$lutRun = 'results_lut/analytic/<实际运行子目录>'

# 只读检查输入、划分与工作量，不计算指标、不创建输出。
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results $lutRun --exp-config configs/lut_quality.json --dry-run

# 正式评估：每次用一个新的输出子目录。
$qualityOut = Join-Path $lutRun 'evaluation/lut_quality_reproduction_01'
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results $lutRun --exp-config configs/lut_quality.json --out $qualityOut

# 比较全量 LUT 时显式指定；它不会保留训练 LUT 的独立留出语义。
$finalLut = Join-Path $lutRun 'normal_lut.npz'
.\.venv\Scripts\python.exe -m experiment.lut_quality --lut-results $lutRun --lut $finalLut --out (Join-Path $lutRun 'evaluation/lut_quality_final_01')
```

默认 `--lut` 始终选训练 LUT，即使目录里存在正式 `normal_lut.npz`。支持 `python experiment/lut_quality/__main__.py --help`。有 manifest 的运行，输出必须位于该运行的 `evaluation/` 下；非空目录拒绝覆盖，当前没有续跑模式。中断后另选新目录，保留旧诊断。

dry-run 的 `source_sample_count` 是原归档总行数，`reference_sample_count/held_out_sample_count` 是划分后的行数；同时给出占用格数、最大格原始行数及 `squared_bin_count_sum=ΣN_b²`。最后一项是粗略工作量提示，不是实际扫描使用的不同坡度数平方和，也不是运行时间预测。

## 8. 输出如何核对

| 文件 | 内容及复现重点 |
|---|---|
| `quality.json` | `format_version=2`，`reference.curves` 只含 `tau/incompatibility`，另有圆心转移的 `holdout`、覆盖摘要、两种状态和警告 |
| `curves.csv` / `curves.png` | 同一容差曲线；CSV 留出字段加 `holdout_` 前缀 |
| `per_bin.npz` | 所有 reference 占用格的稀疏数组，空格不填成零歧义；按 `bin_ids` 升序排列 |
| `top_bins.json` | `format_version=2`，在 `detail_tau` 下先按格内 `A_b` 降序、再按原始样本数降序排名；记录最优圆心及原始归档行号 |
| `top_bins.png` | 排名前四格，每个展示格保留全部原始坡度点；JSON 只交替选取近/远圆心样例，展示限额不参与指标 |
| `bin_heatmaps.png` / `sample_counts.png` | 从占用的 B 节点层级中均匀选最多三层，画真实 R–G 切片；不是平均或最大投影，浅米色表示无样本 |
| `normal_coverage.png` | reference 原始样本的法向覆盖计数，不受 `pose_equal` 加权影响 |
| `manifest.json` | `format_version=2`，实际参数、输入 SHA256、实现文件 SHA256、输入元数据、环境与 `running/completed/failed` 状态 |

令 `B` 为 reference 占用格数、`T` 为容差数，`per_bin.npz` 的主要维度为：

- `[B]`：`bin_ids`、原始 `sample_count`、未归一化 `weight`、`pose_count`、`unique_slope_count`、`original_valid`、倾角范围与法向角 RMS。
- `[B,3]`：RGB 节点 `coordinates`；`[B,2]`：`slope_min/slope_max`。
- `[T,B,2]`：`optimal_center`；`[T,B]`：`incompatibility`；另存 `[T]` 的 `taus`。
- `[18,36]`：`normal_coverage_count`。倾角边界为 `0:5:90` 度，方位角边界为 `-180:10:180` 度，另存两个边界数组。

倾角 `theta=atan(sqrt(p²+q²))`，方位角 `phi=atan2(-q,-p)`。格内 `normal_rms_degrees` 是法向相对于归一化加权平均法向的角度 RMS；它只是诊断信息，不是核心失败比例。`original_valid` 只是保留节点标记，不能据此丢弃格子。`singleton_sample_fraction` 与越界率使用所选样本权重，`sample_count/pose_count` 则是原始计数。

## 9. 小规模复现样例与验收

先运行下面的独立小样例，验证基本定义与权重的影响。它不读取已有运行，也不写结果；保存为临时 Python 文件运行，或在 PowerShell 中通过 here-string 传给 `.\.venv\Scripts\python.exe -`。

```python
import numpy as np
from experiment.lut_quality.metric import evaluate_quality
from utils.normal_mapping import NormalCalibration

table = np.zeros((3, 3, 3, 2), np.float32)
lut = NormalCalibration(table, np.ones((3, 3, 3)),
                        -np.ones(3), np.ones(3), 1.)

# 同格两分支，最有利的一个答案也只能解释一半记录。
colors = np.zeros((10, 3))
truth = np.array([[-1., 0.]] * 5 + [[1., 0.]] * 5)
report, bins = evaluate_quality(colors, truth, np.zeros(10, int), lut, [0., .1, 2.])
row = report['curves'][1]
assert row['incompatibility'] == .5
assert report['curves'][2]['incompatibility'] == 0.
assert bins['sample_count'].sum() == 10

# 同格 9:1 两分支，样本等权时不兼容率为 10%。
colors = np.zeros((10, 3))
truth = np.array([[0., 0.]] * 9 + [[2., 0.]])
poses = np.array([0] * 9 + [1])
report, _ = evaluate_quality(
    colors, truth, poses, lut, [0., .1], weighting='sample')
assert np.isclose(report['curves'][1]['incompatibility'], .1)

# 姿态等权时，两个姿态各占一半，不兼容率变成 50%。
report, _ = evaluate_quality(
    colors, truth, poses, lut, [0., .1], weighting='pose_equal')
assert np.isclose(report['curves'][1]['incompatibility'], .5)
print('LUT quality reproduction examples passed')
```

有本地测试目录时，再运行：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_lut_quality -v
```

该测试文件覆盖非样本圆心、重复点/加权/零容差、相切与跨零角区间、与穷举边界圆对照、分箱舍入、姿态等权、完整姿态隔离、留出未见格、输入签名、精简输出和覆盖保护。若交接给独立克隆，应另附 `tests/test_lut_quality.py`；它目前不在 Git 跟踪范围内。

对真实数据复现时，按以下顺序验收：

1. 对齐具体运行、样本/LUT SHA256、所选 LUT 文件、有效参数与保存的姿态名单。先核对 dry-run 样本数和划分，不先比较图形。
2. 检查 `Σsample_count == reference_sample_count`；按 `weight/Σweight` 对格内曲线加权，应还原 `reference.curves`。
3. 固定输入与权重时，`A` 应在 `[0,1]`，随容差增大不应上升。单样本格的 `A=0` 仅表示当前样本可兼容，不能证明映射唯一。
4. 比较全部容差的 reference/holdout 数值、未见格率与越界率。若另写求解器，最优覆盖量相同但圆心不同，须先核对平局规则再解释留出差异。
5. 比较不同光照或不同 LUT 时，连同姿态、法向覆盖、单样本格比例、颜色范围与 LUT 尺寸一起报告；采样分布变化不能直接当作映射改善。

历史生产报告保持原样。当前版本只输出单值不可兼容率、圆心转移和覆盖诊断；需要精简后的报告时，用原样本和原 LUT 在新的输出目录重新执行本实验，无须重新扫描或拟合 LUT。

本轮验证：`tests.test_lut_quality` 与 `tests.test_light_distribution_experiment` 合计 18 项通过；文中小样例可直接运行。另以固定随机种子 `20261103` 生成 8 组小规模输入，分别使用两种权重，与修改前实现做了 16 组对照：`A` 最大差为 0，保留的逐格数组、最优圆心和留出转移曲线全部一致。没有重算历史生产运行的全量评估。
