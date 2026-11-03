# 解析触觉仿真与深度 GN3 补偿

这里复现 [tactile_sensor_sim](https://github.com/chaoyiL/tactile_sensor_sim/tree/865db6c656548687f2439cafb0dcea76ab01f9ad) 的 analytic 主链：平面或圆柱弯曲曲面 → 标定球压入 → 线性 RGB 渲染 → 虚拟球扫描与法向 LUT → 曲面上的坡度／深度补偿。来源提交和归档指纹见 [SOURCE.json](SOURCE.json)。

## 渲染和虚拟标定

从仓库根目录运行。物理参数位于 [configs/analytic.json](configs/analytic.json)，扫描、拟合和验证参数位于 [configs/calibration.json](configs/calibration.json)。默认使用 CPU，CUDA 渲染按需启用。

```bash
# 平面球压入，同时保存同条件的无压入参考帧
uv run python physical_model/simulate.py --device cpu --reference

# 沿 X 弧长方向弯曲，半径 50 mm；检查配置时加 --dry-run
uv run python physical_model/simulate_curve.py --radius-mm 50 --device cpu --reference

# 平面 LUT 的完整扫描与标定
uv run python physical_model/calibrate_lut.py --device cpu --dry-run
uv run python physical_model/calibrate_lut.py --device cpu

# 相同标定流程也可从曲面渲染入口启动
uv run python physical_model/simulate_curve.py --calibrate-lut --radius-mm 50 --device cpu --dry-run
```

各入口都支持 `--help` 和 `--out`。`--config` 选择物理配置；标定时可另用 `--calibration-config` 选择扫描流程。标定的压入位置、深度由扫描配置决定，`--depth-mm` 可限定单个深度；曲面入口的 `--radius-mm` 表示弯曲半径，独立标定入口用 `--curve-radius-mm` 表示弯曲半径、`--radius-mm` 表示球半径。`--dry-run` 只显示有效参数和工作量，不渲染、不创建输出。

渲染结果默认保存到 `physical_model/render_results/<运行目录>/`，包括以下文件：

| 文件 | 含义 |
| --- | --- |
| `camera_linear_rgb.npy` | 未编码、未裁剪的数值 RGB 观测，形状为 H×W×3 |
| `sensor_irradiance_rgb_w_m2.npy` | 相机传感器各 RGB 波段辐照度，单位 W/m² |
| `radiance_rgb_w_m2_sr.npy` | 曲面辐亮度，单位 W/(m²·sr) |
| `geometry_gt.npz` | 有效像素、世界坐标、光学法向及几何真值 |
| `image.png` | `round(255*clip(RGB,0,1))` 线性预览，不进入标定 |
| `surface.stl`、`metadata.json`、`run.json` | 曲面网格、物理配置、执行状态 |

有 `--reference` 时，两帧分别位于 `indented/`、`reference/`。标定结果默认保存到 `physical_model/lut/<运行目录>/`；每次新建目录，已有非空目录不会被覆盖。运行产物已加入 `.gitignore`。

标定按每个深度的完整接触边界生成压球中心，允许接触区域与表面边缘相切。平面使用接触圆半径；曲面使用三维球面与圆柱的精确材料坐标边界。每个姿态渲染接触图，再在固定材料网格上分别投影和双线性采样接触／参考图后求有符号色差。样本须满足接触内部、正面、第一可见交点、有效四邻域和有限值条件。

训练与验证按完整姿态隔离。LUT 复用本仓库 `utils.normal_mapping` 的构建、填充和三线性查询，以有符号 RGB 色差查询局部坡度 `(p,q)`。每个留出姿态分别检查角度 RMS 和颜色越界比例；默认阈值沿用上游的 20°、5%，节点内 RMS 上限为 5°。

| 标定产物 | 用途 |
| --- | --- |
| `normal_samples.npz` | 样本色差、坡度、材料坐标和姿态索引 |
| `normal_lut_train.npz` | 仅训练姿态拟合的 LUT，用于留出验证 |
| `normal_lut.npz` | 验证通过后用全部样本拟合的正式 LUT |
| `normal_lut_diagnostic.npz` | 验证失败时保存的诊断 LUT，不作为正式发布 |
| `verification/validation.json` | 总体与逐姿态验证、阈值、失败原因 |
| `config.json`、`manifest.json` | 实际生效配置、姿态划分和发布状态 |

实际标定未通过验证时，命令在写完诊断产物后返回退出码 2；`--dry-run` 和成功标定返回 0。

## 补偿验证实验

[configs/compensation.json](configs/compensation.json) 配置测试曲率、独立球姿态、材料网格抽样和 GN3 参数。默认比较平面及半径 200、100、50 mm 的曲面，每个条件使用四个独立测试姿态。实验直接读取选定标定的物理配置快照和材料坐标轴，不使用后来修改的默认物理配置。

```bash
uv run python physical_model/compensate.py --lut-results physical_model/lut/your_flat_run --dry-run
uv run python physical_model/compensate.py --lut-results physical_model/lut/your_flat_run --device cpu

# 本次小扫描未通过发布阈值；显式使用其训练 LUT 做独立测试
uv run python physical_model/compensate.py \
  --lut-results physical_model/lut/verification_20261103_flat \
  --allow-unpublished --device cpu
```

`--lut-results` 优先选择 `normal_lut_train.npz`，也可用 `--lut` 指定同目录下的模型。实验核对线性 RGB、坐标与法向约定，并拒绝与训练姿态重合的测试姿态。失败标定的训练 LUT 需要显式 `--allow-unpublished`，报告保留原标定失败状态。全部样本拟合的诊断 LUT 不作为该实验输入。

测试色差来自与标定相同的图像渲染、材料投影和双线性采样流程。每个条件、姿态比较直接查询平面 LUT、固定位置的坡度 GN3、联合坡度／深度 GN3。三种方法使用同一组有效点；求解失败或秩亏点仍参与指标，诊断另行保存。总体坡度 RMSE 按姿态等权汇总面积加权均方误差后开方。另在同一姿态跨曲率的材料点交集上评分，记录共同点数和原有效点数。

默认产物位于 `physical_model/compensate_results/<运行目录>/`，包括配置与来源快照 `config_snapshot.json`、总体指标 `metrics.json`、逐姿态指标 `per_pose_metrics.csv`、共同域指标 `common_domain_metrics.json`、坡度／深度误差图，以及 `raw/<曲率条件>/<姿态>.npz` 中的逐点观测、真值、预测和求解诊断。`--out` 可指定新运行目录。

补偿配置中的 `output_dir` 相对于所选配置文件所在目录解析；CLI 的 `--out` 相对于当前工作目录解析。默认配置以 `../compensate_results` 指向本包下的结果目录。

## 深度 GN3 的输入和计算

方法依据上游的[平面—切平面 LUT 迁移补偿方法](https://github.com/chaoyiL/tactile_sensor_sim/blob/865db6c656548687f2439cafb0dcea76ab01f9ad/docs/%E5%B9%B3%E9%9D%A2-%E5%88%87%E5%B9%B3%E9%9D%A2LUT%E8%BF%81%E7%A7%BB%E8%A1%A5%E5%81%BF%E6%96%B9%E6%B3%95.md)第 5、6 节。未知量是每个材料点的局部坡度和向内压入深度。设参考位置为 $X_0$，参考几何法向为 $n_0$，正交切基为 $E=[e_1,e_2]$：

$$
X(\delta)=X_0-\delta n_0,\qquad
n_{\rm optical}(\beta)=\frac{-n_0+E\beta}{\sqrt{1+\|\beta\|^2}},\qquad \beta=(p,q)^T.
$$

位置和深度单位为 mm，坡度无量纲。向相机的光学法向为负参考法向的扰动。每个有限光源按余弦幂发射、逆平方衰减、朗伯反射和相机余弦四次衰减构造 $B$：

$$
f(\beta,\delta)=B(X(\delta),n_{\rm optical}(\beta))n_{\rm optical}(\beta)-C_{\rm bg}.
$$

背景和参考面的精确遮挡可见性固定；每轮在当前候选位置更新灯距、灯向、发射／接收活动集、相机增益及其深度导数。坡度 Jacobian 为 $B\,\partial n/\partial\beta$，深度 Jacobian 为 $(\partial B/\partial\delta)n$。

初始坡度来自冻结平面 LUT，初始深度为配置中的固定值，默认 0 mm。每轮用标量 Schur 消元联合求坡度／深度增量；深度受物理区间和步长约束，触边后重解坡度，限制坡度增量后回代深度。候选状态按真实 RGB 目标逐点回溯，固定执行三轮。弱深度列、秩亏、求解失败和拒绝更新保留为诊断。测试球心、真实深度和法向只用于实验评分。

CPU 补偿使用 NumPy float64，复用了模型的光源、几何和遮挡计算；不需要 PyTorch。`compensation_utils` 中的求解器与实验文件读写分开，响应场和求解器可独立调用。

```python
from physical_model.compensation_utils.photometry import build_response_field
from physical_model.compensation_utils.depth import predict_depth_gauss_newton

# coordinates: N×2 参考弧长(mm)；color_delta: N×3 有符号线性 RGB
# beta_lut: N×2 平面 LUT 初始坡度；config: 同一物理条件的配置
field = build_response_field(coordinates, config)
(beta, depth_mm), diagnostics = predict_depth_gauss_newton(
    field, color_delta, beta_lut, initial_depth_mm=0.0, max_depth_mm=2.0,
)
```

## 文件边界和后续实物接入

```text
physical_model/
  simulate.py                 # 平面渲染入口
  simulate_curve.py           # 曲面渲染与 --calibrate-lut 入口
  calibrate_lut.py             # 虚拟标定入口
  compensate.py               # 补偿验证实验入口
  model_utils/                # 几何、光源、辐射度、CPU/CUDA 渲染
  cali_utils/                 # 扫描、材料采样、拟合、验证和可视化
  compensation_utils/         # 连续点响应、联合 GN3 和实验编排
  configs/                    # 可编辑物理与流程配置
  lut/                        # 生成的虚拟标定
  compensate_results/         # 生成的补偿实验
```

当前渲染／扫描支持平面和沿 X 弧长弯曲的圆柱，几何长度单位为 mm。材料坐标 `(s,t)` 表示参考曲面的真实弧长，不能用弯曲后的世界 XY 或像素距离替换。法向补偿链要求 `view_side_z=-1`、`direction_normal=-1`。

该实现为后续实物迁移提供模型和求解接口，尚未接入 `reconstruction/recon.py`。实物链需要同一材料点的 RGB 配对、整体曲面的参考位置／法向／正交切基，以及实测光源、反射率、曝光和相机颜色响应。相机 BT.709 解码得到的线性 RGB 与仿真颜色还需实测辐射标定；不能只修改元数据就把虚拟 LUT 当作现有实物 LUT 使用。

## 依赖和验证

CPU 使用项目现有 NumPy、SciPy、Pillow、Matplotlib。CUDA 渲染代码按需导入 PyTorch；安装可选依赖后显式选择设备：

```bash
uv sync --extra physical-gpu
uv run --extra physical-gpu python physical_model/simulate_curve.py --radius-mm 50 --device cuda:0 --reference
uv run python -m unittest discover -s physical_model/tests -v
```

## 本次数值验证

使用独立上游源码、固定配置及 PyTorch CPU 参考实现进行对照。完整默认扫描尚未运行，CUDA／compiled 路径尚未实际执行；以下为 CPU 检查。

| 检查 | 实际结果 |
| --- | --- |
| 默认 640×480 平面与 R=50 mm 球压入／无压入四帧 | 均完成，结果见 `render_results/verification_20261103_flat/`、`render_results/verification_20261103_curve50/` |
| 平面及 R=200/100/50 mm，压入 0/0.5/1.5 mm，共 12 组渲染 | 辐照度、辐亮度最大绝对差为 0；全部几何真值一致 |
| 平面 140 点、R=50 mm 曲面 133 点的材料采样 | 有效掩膜、色差、坡度和材料坐标逐点一致 |
| 平面／曲面各 48 点的连续响应和解析 Jacobian | 最大绝对差约 $2\times10^{-15}$ |
| 同输入、同参数执行三轮深度 GN3 | 坡度最大差 $5.81\times10^{-13}$，深度最大差 $4.46\times10^{-12}$ mm；关键诊断掩膜一致 |
| 固定上游参考与数值不变量回归 | 15 项通过；另有 8 项现有图像读写测试通过 |

固定参考数据保存在 `tests/fixtures/depth_gn3_reference.npz`，包含上游提交号、输入及未修改上游求解器的预期结果。

已生成的两次虚拟标定均使用 128×96 图像、0.5 mm 材料网格、16³ LUT 和 18 个完整压入姿态，保持原有发布阈值。平面共 2343 样本，留出 RMS 26.29°；R=50 mm 曲面共 2274 样本，留出 RMS 25.10°。两者均超过 20°，分别在 `lut/verification_20261103_flat/`、`lut/verification_20261103_curve50/` 保存训练与诊断 LUT，未生成正式 LUT。

使用上述平面训练 LUT、显式 `--allow-unpublished` 和默认补偿配置，已完成 4 个曲率条件 × 4 个独立球姿态的图像采样实验。结果位于 [compensate_results/verification_20261103_image/metrics.json](compensate_results/verification_20261103_image/metrics.json)。

| 指标域 | 直接平面 LUT | 固定位置坡度 GN3 | 深度 GN3 | 深度 GN3 的深度 RMSE |
| --- | --- | --- | --- | --- |
| 各曲率完整有效域，共 528 点 | 0.4402 | 0.1377 | 0.1201 | 0.2656 mm |
| 跨曲率共同材料点，共 512 点 | 0.4308 | 0.1233 | 0.1034 | 0.2695 mm |

前三列为无量纲坡度向量 RMSE。共同域每个测试姿态有 21、30、35、42 个材料点，在四个曲率条件分别评分。图像分辨率较低、源 LUT 未通过发布阈值，这些数值用于验证迁移和补偿链路，不代表完整扫描或实物重建的精度。
