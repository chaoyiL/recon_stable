# 单图整体与局部重建

`surface_reconstruction/` 提供 marker 检测、跟踪与整体曲面拟合，`get_surface.py` 负责整体重建的文件接口。
`local_recon.py` 提供独立局部重建，`pipeline.py` 串联完整流程，`recon.py` 提供统一命令 `recon`。
`viewer.py` 绘制结果预览，由 `--show` 或 `local-recon --view` 显式打开。

```text
reconstruction/
├── recon.py                  # 单图完整流程 CLI：uv run recon
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

默认输出根目录为 `reconstruction.output_dir`（当前是 `assets/reconstruction`）：

- NPZ：`assets/reconstruction/pc/<图片名>_local_result.npz`。

程序自动创建所需目录，不自动保存检查 PNG。独立 `local-recon` 同样使用这组目录，默认文件名为 `<输入包名>_local_reconstruction`。
NPZ 字段及窗口预览与独立局部重建一致；预览显示三维曲面和法向位移，不显示置信度。
二维位移图按三维曲面行、列曲线的代表性实际长度保持宽高比例，坐标标签仍为材料网格索引。
可用 `--output result.npz` 指定 NPZ 的准确位置，`--show` 在保存后打开预览。
也可使用 `uv run python -m reconstruction.recon`，参数相同。

## 固定输入与处理流程

```yaml
reconstruction:
  reference_image: assets/reference.npz
  material_grid_file: assets/normal_calibration/lut_material_grid.npz
  output_dir: assets/reconstruction
  match_surface_aspect: true
local_reconstruction:
  calibration_file: assets/normal_calibration/normal_lut.npz
```

配置路径相对配置文件解析，命令行路径相对工作目录解析。
参考图、材料网格和 LUT 可分别用 `--reference-image`、`--material-grid`、`--calibration` 覆盖。

程序先在无接触参考图初始化 marker ID，再跟踪当前图并独立拟合当前整体曲面。
参考与当前曲面按同一材料网格评估 XYZ，分别投影到各自图像的 UV；两张图在读取入口统一为线性 RGB、
屏蔽 marker 和饱和像素，再独立采样、相减。色差交给原有 LUT 与局部求解器，直接输出结果。
无需人工合成中间 NPZ，不执行标注或 LUT 训练。当前整体拟合使用有效 marker，不自动排除接触区。

新采集图片使用 `linear_rgb: float32 RGB [0,1]` 的 NPZ，旧相机 PNG/JPG 只在统一读取入口解码一次。
几何缓存 v3 保存 `observed_linear_rgb`，不再把线性图转成 BGR8 落盘。
原图预览直接显示线性值；marker 检测单独适配为原来的 8 位灰度强度，不改变光度计算。

位移沿当前整体曲面的局部法向，单位 mm，输出 `xyz_out = xyz + displacement * normal`。
置信度仍只作诊断，不参与局部方程加权。色差零坡度保护、求解精度沿用 `local_reconstruction` 配置。

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
例如长边 400 点、区域约 15×42 mm 时，网格约为 400×142，具体点数随当前曲面变化。
设置 `match_surface_aspect: false` 可严格使用 observation 配置的行列数。

材料文件可为与 LUT 编号匹配的完整弧长图，也可为已保存的冻结逆采样网格。
图片法向标定的 `fit` 验证通过后，自动从同批 reference 坐标图重新生成
`work_dir/lut_material_grid.npz`，同时发布最终 LUT。网格尺寸取 reference 的 observation 行列数，
保留本批坐标编号、float64 原生参数、st、相机和模板，不沿用旧文件的坐标或尺寸。
单独重跑 `fit` 同样更新网格；reference 不配套或验证失败时保留此前发布的 LUT 和网格。
重建配置的 `material_grid_file` 应指向本批 work_dir 下的该文件。

它提供参考坐标图的逆映射；改变求解分辨率时，在规则 st 上双线性插值原生参数，然后重新计算
两张图各自的 XYZ、UV 和颜色。完整弧长图则直接反解目标 st。两条路径均保留坐标身份，不修改 LUT，
也不缩放已有位移结果；冻结逆网格插值的精度受原网格分辨率限制。
新批次 prepare 后，完成相应 samples 和 fit，再使用同批 LUT 与采样网格。
标定后成像条件变化会影响 LUT 的精度，配套网格只保证坐标与颜色配对一致。

## 独立模块

```bash
uv run get-surface --config config.yaml --image input.png --output overall.npz
uv run local-recon --config config.yaml --input paired_input.npz --output local.npz
uv run local-recon --view assets/reconstruction/pc/local.npz
```

`recon-local` 作为 `local-recon` 的兼容别名保留。

整体曲面和 marker 的调试命令见 [整体重建模块说明](surface_reconstruction/README.md)。
根目录的 `slope_color_audit.py` 是独立离线诊断工具：回放 LUT 坡度、对照色差，统计位移正负交界并输出图、CSV 和 JSON。
它要求同目录中有旧式的整体几何 NPZ、`*_recon_input.npz`、原图和 `*_local_result.npz`；
当前单图流程不保存这些中间文件，因此仅有最终结果 NPZ 时不能直接运行完整审计。该工具不参与重建求解。

## 保留的数据

`assets/normal_calibration/` 保留当前 LUT、训练样本、标注、相关几何缓存、验证结果和冻结材料网格。
`lut_source_images/` 保留从旧标定缓存恢复的 17 张历史标定图和底色图，来源见 `lut_sources.json`；当前 LUT 的配套关系以自身 metadata 为准。
新批次照片保留在 `assets/cali_norm_pics/`，运行时参考图为 `assets/reference.npz`；旧 PNG 仍可显式配置。
相机标定结果也保留。
