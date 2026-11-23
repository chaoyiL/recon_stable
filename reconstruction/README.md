# 单图整体与局部重建

`surface_reconstruction/` 提供 marker 检测、跟踪与整体曲面拟合，`get_surface.py` 负责整体重建的文件接口。
`local_recon.py` 提供独立局部重建，`pipeline.py` 串联完整流程，`recon.py` 提供统一命令 `recon`，`recon_batch.py` 提供五方法入口 `recon-batch`。
`viewer.py` 绘制结果预览；完整重建自动保存图片，也可由 `--show` 或 `local-recon --view` 打开窗口。

```text
reconstruction/
├── recon.py                  # 单图完整流程 CLI：uv run recon
├── recon_batch.py            # 单图五种 LUT：uv run recon-batch
├── pipeline.py               # 整体曲面、材料配对、色差、局部求解
├── get_surface.py            # 独立整体重建 CLI：uv run get-surface
├── local_recon.py            # 局部求解和离线 NPZ CLI：uv run local-recon
├── viewer.py                 # 结果 NPZ 的显示与预览图片导出
└── surface_reconstruction/   # marker、跟踪、整体曲面算法及调试工具
```

```bash
uv run recon --config config.yaml \
  --image assets/cali_norm_pics/capture_20261016_184356_361449.npz
```

默认结果 NPZ 的根目录为 `reconstruction.output_dir`（当前是 `assets/reconstruction`），预览图片固定保存在 `assets/reconstruction/pics`：

- NPZ：`assets/reconstruction/pc/<图片名>_local_result.npz`。
- 预览图片：`assets/reconstruction/pics/<图片名>_local_result.png`。

程序自动创建所需目录。独立 `local-recon` 默认仅保存 NPZ，文件名为 `<输入包名>_local_reconstruction`。
NPZ 字段及窗口预览与独立局部重建一致；预览显示三维曲面和法向位移，不显示置信度。
二维位移图按三维曲面行、列曲线的代表性实际长度保持宽高比例，坐标标签仍为材料网格索引。
可用 `--output result.npz` 指定 NPZ 的准确位置，预览图片始终保存在 `assets/reconstruction/pics/` 下并使用同名 PNG；`--show` 在保存后打开预览。
也可使用 `uv run python -m reconstruction.recon`，参数相同。

## 五种 LUT 批量重建

`recon-batch` 使用与 `recon` 相同的图片输入，一次依次运行 `table`、`mlp3`、`mlp5`、`mlp6`、`mlp8`，每种方法保存一个 NPZ 和一张预览 PNG：

```bash
uv run recon-batch --config config.yaml --image assets/test_pics/3b2.npz
# 同一入口也可通过模块运行：
uv run python -m reconstruction.recon_batch --config config.yaml --image assets/test_pics/3b2.npz
```

结果沿用原来的 `pc/` 和 `pics/` 根目录，每张输入图片集中到 `<图片名>_batch/`：

```text
assets/reconstruction/
  pc/3b2_batch/
    table.npz
    mlp3.npz
    mlp5.npz
    mlp6.npz
    mlp8.npz
  pics/3b2_batch/
    table.png
    mlp3.png
    mlp5.png
    mlp6.png
    mlp8.png
```

NPZ 根目录遵循 `reconstruction.output_dir/pc/`，PNG 根目录固定为 `assets/reconstruction/pics/`。
`--output` 可指定本次 NPZ 目录；预览仍放在 `pics/<图片名>_batch/`。重复运行覆盖同名产物。
参考图和求解参数沿用 `--reference-image`、`--lsmr-atol`、`--lsmr-btol`、`--lsmr-max-iterations`；`--show` 逐方法打开预览窗口。

每种方法从配置的 `normal_calibration.lut_dir/<方法>/` 按版本名选取最新同时具有 `normal_lut.npz` 和 `lut_material_grid.npz` 的版本，包含 `training_only` 候选。
程序启动时读取全部五个模型，并打印所选版本与验证状态。每个模型使用同版本材料网格，配套约定交给单图链路核验。

## 固定输入与处理流程

```yaml
normal_calibration:
  method: table  # table / mlp3 / mlp5 / mlp6 / mlp8
  lut_dir: lut
reconstruction:
  reference_image: assets/reference.npz
  output_dir: assets/reconstruction
  match_surface_aspect: true
contact_surface:
  offset_mm: 7.0
```

配置路径相对配置文件解析，命令行路径相对工作目录解析。
`normal_calibration.method` 同时选择生成方法与重建读取的方法，模型和网格由 `lut/<方法>/current.json`
指向同一个已验证版本。`table` 是原离散表；`mlp3` 输入色差，`mlp5` 输入色差和材料 st，`mlp6` 输入色差和局部曲率三个分量，`mlp8` 同时输入色差、st 和曲率。M6/M8 的曲率取自当前整体基准面，分量顺序为 `k11,k12,k22`，单位为 `mm⁻¹`，与输出坡度使用同一局部切基。
参考图、材料网格和模型可分别用 `--reference-image`、`--material-grid`、`--calibration` 覆盖；显式模型默认使用其同目录网格。

程序先在无接触参考图初始化 marker ID，再跟踪当前图并独立拟合参考和当前的 marker 曲面。
有效重建轮廓取黑框内沿围出的胶皮区域，不取最外圈 marker。参考图将黑框轮廓反投影到接触层，
建立包含该轮廓的冻结原生参数域；原 marker 域外沿边界切向延拓，再沿单位法向偏移 `contact_surface.offset_mm`。
后续帧使用同一材料参数域，分别按各自黑框 mask 裁剪，不把 marker 搜索 mask 的侵蚀用于重建。
参考和当前接触面按同一材料网格评估 XYZ，分别投影到各自图像的 UV；两张图在读取入口统一为线性 RGB、
屏蔽 marker 和饱和像素，再独立采样、相减。色差与模型需要的位置/曲率交给所选法向模型与局部求解器，直接输出结果。
无需人工合成中间 NPZ，不执行标注或 LUT 训练。当前整体拟合使用有效 marker，不自动排除接触区。

新采集图片使用 `linear_rgb: float32 RGB [0,1]` 的 NPZ，旧相机 PNG/JPG 只在统一读取入口解码一次。
独立整体几何 v3 和图片标定几何缓存 v5 均保存 `observed_linear_rgb`；标定缓存额外记录接触面偏移和 `domain_kind`。
原图预览直接显示线性值；marker 检测单独适配为原来的 8 位灰度强度，不改变光度计算。

位移沿当前接触面的局部法向，单位 mm，输出 `xyz_out = xyz + displacement * normal`。
置信度仍只作诊断，不参与局部方程加权。色差零坡度保护、求解精度沿用 `local_reconstruction` 配置。

黑框几何外沿同时施加零局部位移和零局部坡度。离散求解将外边界及紧邻内层固定为零位移，
使外沿向内的一格单侧差分为零；固定环的输出坡度也为零，`boundary_mask` 记录这些固定点。
这些约束相对当前整体接触面定义，不改变整体弯曲。marker 等内部光度孔洞仍由邻域推断，不作为固定边界。

正负色差的保护阈值可分别设置，均填写幅值：

```yaml
local_reconstruction:
  zero_color_protection:
    inner_radius: 0.08
    outer_radius: 0.15
    negative_inner_radius: 0.03
    negative_outer_radius: 0.08
```

正方向在 0.08 以内归零、0.15 起恢复完整坡度；负方向在 -0.03 以内归零、-0.08 及更负时恢复完整坡度。
各自过渡区使用 smoothstep，两侧响应取较大值统一缩放 LUT 坡度；RGB 各通道可以同时有正值和负值。
省略负方向值时继承正方向配置；四个半径均为 0 时关闭整层保护。
修改这组推理参数只需重跑 `recon`，不需要重新计算 LUT。

求解网格默认匹配当前整体曲面的实际长宽。长边点数取 `material_coordinates.observation`
的行列数较大值；用三维行、列曲线长度的中位数估计宽高，再按相同物理步长确定短边点数。
例如当前黑框域的 `3b3` 网格为 400×155，具体点数随当前曲面变化。
设置 `match_surface_aspect: false` 可严格使用 observation 配置的行列数。

材料文件可为与 LUT 编号匹配的完整弧长图，也可为已保存的冻结逆采样网格。
图片法向标定的 `fit` 验证通过后，自动从同批 reference 坐标图重新生成
`lut/<方法>/<版本>/lut_material_grid.npz`，同时发布该版本模型。网格尺寸取 reference 的 observation 行列数，
保留本批坐标编号、float64 原生参数、st、相机和模板，不沿用旧文件的坐标或尺寸。
单独重跑 `fit` 同样更新网格；reference 不配套或验证失败时保留此前发布的 LUT 和网格。
默认通过同一版本指针读取这两个文件，不再分别配置固定的模型和材料网格路径。

它提供参考坐标图的逆映射；改变求解分辨率时，在规则 st 上双线性插值原生参数，然后重新计算
两张图各自的 XYZ、UV 和颜色。完整弧长图则直接反解目标 st。两条路径均保留坐标身份，不修改 LUT，
也不缩放已有位移结果；冻结逆网格插值的精度受原网格分辨率限制。
新批次 prepare 后，完成相应 samples 和 fit，再使用同批 LUT 与采样网格。
标定后成像条件变化会影响 LUT 的精度，配套网格只保证坐标与颜色配对一致。
修改 `contact_surface.offset_mm` 后需要从 `prepare` 重新执行图片法向标定；缓存、LUT 和材料网格记录偏移量，重建会拒绝不匹配的产物。

黑框域使用坐标图 v2 和 `domain_kind=black_frame_tangent_v1`。旧 marker 范围的坐标图、缓存、LUT 和网格不能混用，
需保留原图标注后重新执行 `prepare`、`samples`、`fit`。样本与几何缓存在 `assets/normal_calibration/black_frame/`，
所有模型版本及配套网格集中在 `lut/`；历史模型的精确归档在 `lut/legacy/`。
原生参数与冻结网格均保留 float64，避免非整值边界因 float32 舍入落到域外。
扩域部分没有 marker 直接约束，其几何属于边界切向外推；结果 `extrapolated_mask` 标识这部分有效点，
不表示该区域的色差—坡度关系已经被标定。最终轮廓也受黑框分割精度影响。

## 独立模块

```bash
uv run get-surface --config config.yaml --image input.png --output overall.npz
uv run local-recon --config config.yaml --input paired_input.npz --output local.npz
uv run local-recon --view assets/reconstruction/pc/local.npz
```

`recon-local` 作为 `local-recon` 的兼容别名保留。
独立 `get-surface` 默认输出 marker 下曲面，指定 `--contact-surface` 时按配置输出法向偏移后的整体接触面。`surface_reconstruction/` 输出 marker 下曲面。
`local-recon` 接收已经生成的接触面几何，输入与 LUT 均需声明一致的 `contact_offset_mm`；不会在读取外部 NPZ 时再次偏移。

`contact_surface.curvature_smoothing_half_width_mm` 控制 marker 拟合域与切向外推域接缝向每侧延伸的曲率平滑距离，当前为 2 mm。高斯标准差取半宽的四分之一，余弦权重在半宽处衰减到零。设为 `0` 时输出原始曲率。

`get-surface` 在 observation 网格上输出 `curvature_grid_per_mm`、`curvature_valid_mask` 和 `curvature_smoothing_half_width_mm`。`recon` 在求解网格上输出同名整体接触面曲率字段，并保存配套的 `st` 和 `coordinate_chart_id`。曲率分量按 `k11,k12,k22` 排列，单位为 `mm⁻¹`，供光场模型使用。XYZ、UV、局部求解和已有法向 LUT 沿用各自的几何与标定约定。

学习光场的完整流程可通过 `uv run python -m optical_learning.generation_03.run --run-name my_run` 执行。它按 `normal_calibration.background_contact_exclusion_margin_px` 采样背景、训练光场，再标定配套 Table 并重建 `3b2`。参数为原始图像中的像素半径，当前为 40 px。法向标定的背景检查也使用该值。

整体曲面和 marker 的调试命令见 [整体重建模块说明](surface_reconstruction/README.md)。
根目录的 `slope_color_audit.py` 是独立离线诊断工具：回放 LUT 坡度、对照色差，统计位移正负交界并输出图、CSV 和 JSON。
它要求同目录中有旧式的整体几何 NPZ、`*_recon_input.npz`、原图和 `*_local_result.npz`；
当前单图流程不保存这些中间文件，因此仅有最终结果 NPZ 时不能直接运行完整审计。该工具不参与重建求解。

## 保留的数据

`assets/normal_calibration/` 保存标注、几何缓存和样本。`lut/` 保存所有版本的离散表、MLP、配套网格、验证报告与效果对照，整目录 Git 忽略。
历史迁移清单 `lut/migration_manifest.json` 记录原路径、归档路径和 SHA-256；旧 LUT 的数组值保持不变。
`lut_source_images/` 保留从旧标定缓存恢复的历史标定图和底色图，来源见 `lut_sources.json`；LUT 的配套关系以自身 metadata 为准。
新批次照片保留在 `assets/cali_norm_pics/`，运行时参考图为 `assets/reference.npz`；旧 PNG 仍可显式配置。
相机标定结果也保留。
