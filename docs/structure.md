# 当前仓库结构与主链路

核对日期：2026-10-16。压球图片标定的算法和文件契约见 [cali_norm_plan.md](cali_norm_plan.md)。

## 1. 主链路与坐标约定

```text
底色图 → 黑框内部轮廓/marker → 参考整体曲面 → 冻结参考弧长图
                                               ↓
压球图 → 从参考图恢复 ID → 只跟踪 → 标注接触边界 → 排除接触 marker 后拟合
                                               ↓
规则 st → 同一个参考弧长图反解 → 原生 a,b → 当前/参考 XYZ 和各自 UV
                                               ↓
两张原图分别转线性 RGB、分别采样 → 有符号曲面域色差
接触边界 + 最终整体曲面 + 真实球半径/侧 → 球心 → 同 UV 射线的固定球冠 → 法向/坡度
                                               ↓
统一有效索引配对 → 整张图片留出验证 → 通过后发布最终 LUT
```

- `parameters / coordinates / material_coordinates`：模板定义的原生材料参数 a,b，单位 mm；双三次曲面始终表示 `Q(a,b)`。
- `st`：参考曲面两族三维材料曲线的归一化累计弧长，无量纲。它通常相互耦合，不能代替物理距离或直接传入 bicubic basis。
- `uv`：含相机畸变的原图像素坐标。当前图与参考图使用各自的 UV。
- 三套网格 `geometry / lightfield / observation` 使用同一张参考弧长图，只改变输出分辨率。`lightfield` 仅为保留的接口名称，不生成背景光场。
- 局部基统一为最终规则 st 网格的列、行方向：`frame_convention=st_grid_columns_rows`。

弧长图从 129² 原生网格开始积分，加密至 257²、513²、1025²；收敛和每个 cell 的正 Jacobian 均通过才接受。正向插值和二维 Newton 逆解共用同一表达式。坐标图 UUID 随文件保存/加载保持不变，当前形变不会重新定义材料身份。

## 2. 黑框区域与整体重建

`get-surface` 和图片标定均直接使用 `markers.find_inner_region` 返回的 `InnerRegion(roi, contour, mask)`，不初始化独立分割后端。mask 填充实际内部轮廓，包含黑色 marker；`DETECT_X_MIN / DETECT_X_MAX` 仅将 marker 搜索框限制在原图横坐标的闭区间内，不裁剪胶皮 mask。两个常量可在 `markers.py` 中调整，输出的十字中心和连通域仍使用原图坐标。旧 `find_inner_roi` 仍返回矩形。

`SurfaceReconstructionSession.process(frame, fit=False)` 只推进跟踪，不拟合、不累计拟合失败。`fit_observations` 读取已有观测，按接触排除区修改 `marker_fit_valid`，保留跟踪位置供光度屏蔽。每张独立照片重新从同一底色图初始化；重复调整标注使用同一缓存。

`SurfaceReconstructionProvider` 提供：

| 方法 | 用途 |
|---|---|
| `reconstruct` | 普通整体重建，默认使用当帧黑框 mask |
| `track_frame` | 缓存观测，不做曲面拟合 |
| `fit_tracked_frame` | 对缓存观测排除接触区后拟合并生成三套网格 |
| `sample_coefficients` | 评估已有系数，供参考预览和缓存回放 |
| `bind_coordinate_chart` | 绑定已建立的参考弧长图 |
| `reset(clear_coordinate_chart=False)` | 重置跟踪，默认保留材料坐标身份 |

参考初始化要求完整 48 点网格。最终整体拟合沿用 provider 的覆盖、重投影和度量门槛；ambiguous 不接受。

```bash
uv run get-surface --image input.png --output overall_surface.npz
uv run get-surface --image input.png --output overall_surface.npz \
  --coordinate-reference assets/normal_calibration/reference_coordinates.npz
```

未指定参考图时，首个有效曲面建立坐标图，另存为 `<output_stem>_coordinates.npz`。本批次局部重建应复用标定时的坐标图。

## 3. 图片标定命令

保留 `calibrate-norm` 命令。默认 `input_mode: samples`，外部两数组输入继续可用；图片工作流需要在 `config.yaml` 填写本次数据：

```yaml
normal_calibration:
  input_mode: images
  images: ["assets/cali_norm_pics/*.png"]
  reference_image: assets/reference.png
  sphere_radius_mm: 10.0          # 示例；必须换成本次真实半径
  sphere_side: positive_normal    # 或 negative_normal，须明确球心相对局部基准法向的侧
  boundary_model: zero_boundary_displacement
  work_dir: assets/normal_calibration
  annotations_file: assets/normal_calibration/manual_regions.yaml
  input_file: assets/normal_calibration/normal_samples.npz
  output: assets/normal_calibration/normal_lut.npz
```

`zero_boundary_displacement` 假定标注的接触边界三维点同时位于最终整体曲面与球面；可见球轮廓或普通 ROI 不具备这个语义。图片路径、底色、真实半径和球心侧不会从旧标注推断。路径相对配置文件解析。

```bash
uv run calibrate-norm --config config.yaml --stage prepare
uv run calibrate-norm --config config.yaml --stage annotate
uv run calibrate-norm --config config.yaml --stage samples
uv run calibrate-norm --config config.yaml --stage fit
# 图片模式省略 stage 时顺序执行 all。
```

| 阶段 | 行为 |
|---|---|
| `prepare` | 检测参考/当前黑框、参考完整 ID 初始化、独立跟踪、保存可回放缓存；不要求球半径 |
| `annotate` | 对缺标注图片打开三面板编辑器；accepted/skip 默认复用，`--edit` 重新编辑 |
| `samples` | 无界面读取缓存与 accepted 标注，重新拟合最终曲面，生成颜色/球面标签和逐图报告 |
| `fit` | 读取聚合样本及同批 reference 几何缓存；训练、整图留出验证，发布最终 LUT 和重新生成的采样网格；不读取原图、不创建 GUI |

CLI 还可使用 `--images`、`--reference-image`、`--sphere-radius-mm`、`--sphere-side`、`--annotations` 和 `--output` 覆盖配置。`--input` 选择外部样本模式，与 `--images` 互斥；默认阶段为 fit。

```bash
uv run calibrate-norm --input external_samples.npz --output external_lut.npz
```

只有 `color_residual_samples(N×3)` 和 `slope_samples(N×2)` 的外部样本输出标记为 `unvalidated_external_samples`。缺少新坐标声明的旧 LUT 不能直接用于 `reference_arclength` 输入。

### 标注操作

三个面板显示原图与 marker 屏蔽、曲面色差与 sample mask、整体曲面深度预览或球面法向/拟合误差。直接在中间色差图上编辑绿色接触轮廓和红色圆心，左侧原图同步显示轮廓供对照；鼠标编辑只作用于中间面板。进入标注即显示基于参考曲面 UV 的近似色差，标记为 `Reference UV colors`；按 `g` 后改用重拟合曲面的 UV，标记为 `Final UV colors`。面板列出有效颜色点和配对样本数。色差图的显示缩放及面板偏移显式转换回原图像素坐标。

右侧在球面法向尚未算出时显示整体曲面的相机深度 `Z`（mm）：按当前有效区域的最小、最大 Z 映射为 VIRIDIS 颜色，深紫色较近、黄色较远；每张图单独归一化，不能跨图直接比较颜色。`Reference overall depth` 使用准备阶段的参考曲面，`Final overall depth` 使用按当前标注重拟合的整体曲面。球面拟合成功后切换为 `Sphere normals`：RGB 分别编码相机坐标系的 `nx / ny / nz`，分量从 `[-1,1]` 映射到 `[0,255]`；黑色为未参与配对的样本区域。该图来自球面几何标签，不是 LUT 预测、置信度或局部位移。右图使用材料采样网格行列，不是原图像素坐标。编辑圆后清除旧球面标签，恢复参考深度预览；按 `g` 重新计算。

标注窗口使用 Matplotlib 的 PyQt5 画布，禁用导航工具和画面缩放。三栏显示宽度依次为 640、960、280 像素，图像区高度为 720 像素；左侧原图和中间色差图等比例显示，右侧预览紧凑排列，说明文字在上方换行。说明文字和状态栏采用 22 像素字号、30 像素行距，使用独立字体图层按窗口实际分辨率绘制；不再先把小字画进源图后随图像插值放大。文字按实际字宽换行，底部预留四行状态区。窗口初始大小适配屏幕，可手动调整；鼠标位置仍按固定图像坐标转换，滚轮仅修改中间面板内的标注圆半径。保存的标定预览使用相同布局，文字直接在最终 PNG 分辨率绘制。首次安装新增 GUI 依赖时运行 `uv sync`。

使用 Conda 或其他已激活环境时，依赖必须安装到实际运行程序的 Python 中；`uv sync` 默认管理项目 `.venv`，不会自动同步其他环境。可用 `python -c "import sys; print(sys.executable)"` 确认解释器。在该环境安装 PyQt5 后，可直接运行 `python -m calibrate.calibrate_norm --config config.yaml --stage annotate` 继续标注，无需重新 prepare。

中间面板按当前预览几何的 UV 将曲面色差及 sample mask 映射回原图像素坐标。各面板分别等比例缩放，鼠标位置按中间面板的偏移与缩放换算回原图，保存的轮廓仍使用原图像素坐标。投影外、无效颜色和 marker 区域显示为黑色。该反投影仅用于显示，训练仍使用材料网格中的原始色差和有效 mask；编辑轮廓时复用色差预览，重算几何后刷新。

| 操作 | 行为 |
|---|---|
| 在中间色差图拖动圆内部 / 圆周 | 平移圆心 / 调整半径，始终保持圆形且不超过物理尺寸上限 |
| 中间色差图内使用鼠标滚轮 | 在物理上限内调整圆半径，不缩放画面；其他面板和空白区域不响应 |
| `c` 后点击中间色差图 | 设置圆心，同时作为接触区内部代表点 |
| `z` | 撤回当前图片的上一步编辑，可连续撤回；完整拖动算一步 |
| `g` | 用当前标注重新计算最终几何、背景检查和球面标签 |
| `Enter` | 重算通过后接受并继续 |
| `x` | 跳过图片 |
| `Esc` 或关闭窗口 | 保存 draft 并退出；再次运行 annotate 续标 |

标定默认使用原图像素空间中的圆形接触边界，`shape: circle`，`semi_axes_px` 的两个数值均为半径；拖动和缩放都不能改变为椭圆，`a / d / p` 在圆形模式下不生效。球面标签继续通过相机投影和真实球半径计算。

标注圆直径受真实球尺寸限制。这里“最高深度”指有效整体曲面的最大正相机深度 `Z_max`（mm），不是离相机最近的点。令球半径为配置中的 `R`（mm）、标注圆心为原图像素 `c`，将圆心相机射线与 `Z=Z_max` 平面相交得到 `C`。在该平面投影球的最大截面：对 128 个均匀角度计算 `q(theta) = C + R * (cos(theta), sin(theta), 0)`，上限为 `r_max = min ||project(q(theta)) - c||`（pixel）。`project` 使用相机内参和畸变；无畸变时简化为 `D_max = 2 * min(fx, fy) * R / Z_max`，圆形标注采用较小方向的投影尺度。这个尺寸约束使用平行于成像平面的最大截面，不替代后续球面拟合验收。

初始使用 prepared 曲面，重算后使用最近一次得到的最终曲面；移动圆心时重新计算畸变对尺寸的影响。面板显示当前直径和最大直径。新圆、重新打开的旧圆、拖动、滚轮与撤回均限制在上限内。若重算后上限变小，自动收缩并清除过期预览，再次重算通过才能接受；没有有效正深度时报错，不使用默认上限。未打开的 accepted/skip 标注保持不变。

旧椭圆或多边形标注仍可读取；在 annotate 中重新打开该图片时，保留圆心位置并以等面积圆作为编辑初值（多边形转换的最小半径为 2 px），重新确认后保存。已接受或跳过的图片需用 `--stage annotate --edit` 打开；不会批量改写未打开的旧标注。

撤回支持圆心平移、半径调整、滚轮调整半径和中心设置。撤回后按 `g` 重算预览；中心选择尚未点击时，`z` 先取消选择。历史仅保留在当前图片本次编辑过程中。

标注 YAML v2 使用统一 `image_root` 下的完整相对图片路径，支持不同目录同名图片。标注语义固定为 `contact_boundary`。修改标注后从 samples 重跑；相机、模板或参考坐标图变化后重新 prepare。

重算未通过时，保留已经计算出的色差和几何预览，并标记 `REJECTED`。标注模式下，即使背景检查失败，也会继续尝试计算球面法向供诊断；球面失败时右侧显示最终整体深度和失败原因。窗口底部及终端均输出失败详情，标注保存为 draft，`Enter` 不会接受失败图片。诊断预览不能导出为训练样本，原有验收阈值保持不变。

`background_mismatch` 消息同时显示实测值和阈值；需检查接触边界、参考图及拍摄条件。修改轮廓或撤回后，清除上次重算结果并恢复参考曲面下的近似色差，按 `g` 再次重算。鼠标编辑产生的 NumPy 数值在保存时转换为普通 YAML 数值。

## 4. 颜色、球面标签与 LUT 验证

两张原图各自建立 `surface_mask & ~marker_image_mask & unsaturated`。marker mask 复用检测的组件形状；失检或匹配不唯一时使用参考尺寸与跟踪位置保守屏蔽。额外膨胀宽度为 1 px（3×3 核），标定与 `reconstruction/get_surface.py` 共用此默认值。原始 BGR8 任一通道为 255 即排除。

标定加载几何缓存时，从缓存原图、marker 位置及参考尺寸重建光度屏蔽 mask，使调整后的膨胀宽度直接用于 annotate 和 samples；不重新拟合整体曲面或建立坐标图，也不在读取时改写缓存文件。重跑 samples、fit 可更新训练样本和 LUT。已导出的普通整体重建或 recon 输入 NPZ 仍保留其原有 mask，需重新导出才能采用新宽度。

先转线性 RGB，再分别按各自 UV 做严格四邻域双线性采样，最后计算 `observed-reference`。保留三通道正负号，不给 marker 补色。几何 mask 保留 marker 材料位置，光度 mask 表示两侧颜色都有效。

边界按 128 点采样；标签直接使用标注圆盘，不缩小半径或内缩边界，外扩 20 px 仅用于排除整体拟合和背景统计。UV 三角形提供原生参数初值，再在归一化原生参数中优化反投影。边界至少 96 点有效且连续缺失不超过 32 点；多解拒绝。

固定真实半径，用带鲁棒损失的优化拟合球心。先在中心射线选择唯一球冠分支，再将 minus/plus 分支锁定整张图片。每个颜色点沿同一个当前 UV 的射线求真实球面交点，法向转换成共用局部基的坡度。

最终只用同一个 sample mask 提取颜色、坡度、st、原生参数和图片索引。

整图验证按图片 key 排序后用种子 42 打乱，默认留出 20%，至少一张验证图、一张训练图。颜色范围、节点和缺节点补全仅使用训练组。验证直接调用原始 LUT 插值，不应用推理的零色差保护或无接触覆盖。默认要求每张验证图角度 RMS ≤5°、原始色差超训练范围比例 ≤0.05。

先写训练 LUT 和验证 JSON；全部通过后才用全部 accepted 样本重新拟合最终 LUT。同时加载同批 `reference_surface.npz` 和其 `reference_coordinates.npz`，按 reference 的 observation 行列数在规则 st 上重新反解原生材料坐标，生成 `lut_material_grid.npz`。不读取或沿用已有采样网格，单独运行 `fit` 也会重新生成。

发布前检查 reference 与 LUT 的坐标编号、参考图和坐标约定一致，并先完整写好两个临时文件，再分别替换最终 LUT 和采样网格。验证、reference 检查或临时文件写入失败时保留已发布的两个文件。旧两数组外部样本没有 reference 几何，只生成标记为未验证的 LUT，不更新采样网格。

图片 `fit` 成功后自动在 `work_dir/visualization/` 生成本批结果；已有标定也可独立补生成：

```bash
uv run python -m calibrate.visualize_norm --config config.yaml
```

验证失败的诊断 LUT 也可通过 `--calibration PATH` 生成全部图表，但须附带失败验证报告及同名 `_train.npz`。图表统一标记 `FAILED | diagnostic LUT; not published`，`summary.json` 记录验证状态、报告、有效节点数和保留样本数；该操作不发布 LUT，也不更新重建使用的采样网格。

程序读取全样本 LUT、同名 `_train.npz`、聚合样本及配套 `frames/overall` 缓存，不重新训练或打开 GUI。可用 `--calibration`、`--samples`、`--work-dir`、`--output-dir` 覆盖路径。输入须为通过验证的图片 LUT，或附带失败报告的诊断 LUT；旧两数组外部样本没有空间图和独立验证信息。

不良映射图使用拟合及剔除前的球面标签 p/q：对所有无序样本对（包含同图与跨图），若有符号线性 RGB 欧氏距离 ≤色差阈值，且公共局部坐标约定下 `normalize([-p,-q,1])` 的法向夹角 >角度阈值，将两端样本都标红。默认阈值分别为 0.01 和 20°；通过 `--color-distance-threshold` 和 `--normal-angle-threshold-degrees` 调整。一个点只要参与任意冲突对即计为不良点；点数与配对数分别记录，不只查询最近邻。图中 q 轴向下增大，与参考图一致，p/q 数值符号不变。

使用原始标注圆盘重新标定的 deep 结果位于 `assets/normal_calibration_comparison/deep_full_disk/`，原 deep 结果保留作对照。重新绘图示例：

```bash
uv run python -m calibrate.visualize_norm \
  --calibration assets/normal_calibration_comparison/deep_full_disk/normal_lut.npz \
  --samples assets/normal_calibration_comparison/deep_full_disk/normal_samples.npz \
  --work-dir assets/normal_calibration_comparison/deep_full_disk \
  --color-distance-threshold 0.01 --normal-angle-threshold-degrees 20
```

| 输出 | 内容 |
|---|---|
| `color_difference/000000.png` | 原图像素位置的有符号色差预览及线性 RGB 三通道色差；预览为 `0.5 + 2 * delta`，黑色表示无效 |
| `pq_mapping/000000.png` | 最终 LUT 查询得到的 p、q、坡度模长；球面真值 p、q 和仅在配对样本区计算的法向角误差 |
| `color_pq_mapping.png` | 以三个 RGB 色差通道为坐标、分别以预测 p/q 着色的三维映射示意；最多均匀显示 6000 个样本 |
| `lut_slices.png` | 分别固定 ΔR、ΔG、ΔB 为 0 的完整 LUT p/q 切片，包含补全节点 |
| `calibration_summary.png` | 最终 LUT 的拟合散点和角误差分布，以及训练组 LUT 在留出图片上的 p/q 散点和逐图 RMS |
| `pq_ambiguities.png` | 球面标签的 p/q 分布，不良映射点标红；大坡度数据另附中央区域放大图，仅输出 PNG |
| `ambiguities.npz` | 完整样本色差与坡度、图片索引、不良点 mask、冲突对索引、逐对色差距离和法向夹角、两个阈值 |
| `summary.json` | 本批图片索引、完整图片 key、图像路径及拟合/独立验证统计 |

p/q 是局部正交切向基下的无量纲坡度，所有预测都调用原始 LUT，不应用重建阶段的零色差保护或无接触覆盖。逐图 p/q 图展示最终全样本 LUT；独立留出验证使用 `_train.npz`，与 `validation.json` 的分组保持一致。编号按聚合样本中的 `image_paths` 顺序，完整对应关系见 `summary.json`。

空间图通过现有 UV 映射回原图，并裁剪到可见曲面附近。色差及 p/q 色标在本批图片间一致；角误差图以本批 RMS 阈值作为显示上限，超出值使用顶端颜色并由色条箭头标明，不参与重新判定有效性。图像回投只用于显示，统计直接在原始材料网格样本上计算。旧批次缓存按编号排除；输入不配套或生成失败时保留原报告。报告目录内出现无关文件时拒绝覆盖。可视化失败会明确报错，已发布 LUT 的验证状态保持不变。

`calibrate.analyze_ambiguities` 进一步分析不良点的色差符号、强度、倾角、图片及接触圆内位置，比较图像域的局部法向平滑性，并检查每个不良点颜色邻域内的坡度分支：

```bash
uv run python -m calibrate.analyze_ambiguities \
  --work-dir assets/normal_calibration_comparison/deep_full_disk \
  --color-distance-threshold 0.01 --normal-angle-threshold-degrees 20
```

默认输出到本批目录的 `ambiguity_analysis/`：PNG 展示统计规律、原图中的不良点、局部 p/q 分支、外围分支支撑及改进实验；`analysis.json` 记录定义、源文件哈希和完整统计，`samples.csv` 保存逐点诊断。颜色图统计不良点比例，不使用红点数量代替风险；另记录每个近色邻居的冲突比例，避免混淆样本密度与歧义。

局部颜色球以不良点为中心、RGB 距离为半径，因此不同球可以重叠，分类数不代表独立板块数。原颜色球的连通分量描述保留在 `neighbourhood_classification`：法向图用 ≤10° 的边连接样本，并记录 7.5°、10°、15° 连边阈值的敏感性；其中少于 6 点只描述原颜色球内的数量。正式的“主支撑簇／多分支重叠”分类使用下面的外围扩展证据，不以原颜色球点数停止分析。

少于 6 点只描述原颜色球内的数量，不停止分析或删除样本。对全部冲突对，继续以两端色差中点查询 2、3、4 倍色差阈值的外围邻域，并排除两端各自的原颜色球，防止冲突点和原邻域给自身提供支撑。分别记录外围法向与两端法向相差 ≤原法向阈值一半的点数、图片数、其他图片数、最近色差距离和外围最小法向差。`branch_continuation.csv` 保存全部逐对、多尺度证据，`branch_continuation.png` 展示两支都有外围支撑和仅一支有支撑的实例。此处用已知端点法向追踪候选分支，不是无偏预测验证；外围没有支撑也不自动等于错点，可能是覆盖不足或分支变化快。

扩展分类默认使用 3 倍色差阈值的外围半径。设两支的外围支撑点数为 A、B：总支撑大于零且 `max(A,B)/(A+B) ≥ 0.8` 时记为 `main_support`（主支撑簇），否则两支都有支撑时记为 `multiple_branches`（多分支重叠）；均无支撑才记为 `no_outer_support`。没有最少 6 点或最少 3 点的门槛。`--dominance-fraction` 可调整主支撑比例；报告同时比较 0.7、0.8、0.9，以及三个扩展半径。支撑锥的半角为原法向阈值的一半，而冲突对法向差严格大于原阈值，因此两锥不会重复计数。

一个不良点可能与多个点冲突：只要与任一冲突点形成多分支重叠，逐点分类就记为多分支重叠；其余存在主支撑关系的点记为主支撑簇。每个点只计数一次，同时保留混合关系及强支／弱支角色。`branch_classification.png` 展示色差与 p/q 分布和扩展半径敏感性；`branch_classification_pairs.csv`、`branch_classification_points.csv` 分别保存逐对与逐点结果，完整统计位于 `analysis.json` 的 `branch_continuation.expanded_classification`。原球面标签、LUT 和样本均不修改。支撑比例反映采样密度，不代表标签正确概率；报告另记图片数判据是否一致、是否有多图支撑及跨尺度稳定性，不能仅据主支撑分类删除弱支。

改进实验比较现有 LUT、仅颜色的单位法向插值、加入材料位置的插值，以及删除全部不良训练点后重建 LUT。插值在 `[ΔR,ΔG,ΔB,weight*s,weight*t]` 中查近邻，按距离平方倒数加权单位局部法向后归一化；st 是无量纲材料位置，不作为物理距离。权重与邻居数仅由原训练组内部逐图留出选择，再评估原来的独立留出图片；删点也只使用训练组配对。`remedy_predictions.npz` 保存留出预测，`experimental_position_model.npz` 保存训练组的位置条件原型。原型尚未接入重建，分析不修改已发布的 LUT、配置或已有重建结果。验证仍基于球面标签，无法单独判定真实光学多值和标签偏差。

`calibrate.compare_norm` 分别标定 `assets/cali_norm_pics(deep)` 和 `assets/cali_norm_pics(shallow)`，沿用当前底色和各图已有标注。两组使用同样的 LUT 参数，结果隔离保存在 `assets/normal_calibration_comparison/{deep,shallow}/`，不覆盖重建所用模型。

```bash
uv run python -u -m calibrate.compare_norm --config config.yaml
```

比较包括逐图留出、各取 8 张训练图的 10 轮共同测试、按球面标签倾角分段的误差、样本支撑、异图近色标签一致性及颜色扰动敏感度。颜色距离统一使用绝对线性 RGB 欧氏距离；近色标签比较在公共局部坡度基下进行，不直接比较不同曲面点的相机法向。`comparison.json` 包含指标定义和全部统计，两个 CSV 保存逐图及逐轮结果，三个汇总 PNG 展示对照。`--reuse-samples` 只在输入文件和配置哈希一致时复用已生成样本。

在上述比较完成后，可运行混合标定实验：

```bash
uv run python -u -m calibrate.compare_mixed_norm --config config.yaml
```

全部图片在同一参考坐标图下重新生成样本，逐图确认数值与单组标定一致，混合 LUT、配套网格和可视化保存在比较目录的 `mixed/`。实验沿用原比较的 10 轮留出图：`mixed8` 使用 4 张 deep 和 4 张 shallow，与两种单组 8 张模型比较；`mixed16` 使用留出图之外的两组各 8 张训练图，单独评估额外数据的收益。测试图始终排除在所有模型的训练集之外。`mixed_comparison.json`、`mixed_common_test_rounds.csv` 和 `mixed_common_tests.png` 保存完整指标、逐轮结果和对照图；其中混合逐图留出使用 20 张训练图，不能当作控制训练数量的实验。原标定及单组比较产物保持不变。`--reuse-samples` 依据源文件哈希复用混合样本。

## 5. 输出与缓存契约

```text
assets/normal_calibration/
  manifest.json
  reference_surface.npz
  reference_coordinates.npz
  prepared/000000.npz
  overall/000000.npz
  frames/000000.npz
  manual_regions.yaml
  normal_samples.npz
  normal_lut_train.npz
  normal_lut.npz
  lut_material_grid.npz
  lut_sources.json
  lut_source_images/
  verification/000000.png
  verification/report.csv
  verification/validation.json
  visualization/color_difference/000000.png
  visualization/pq_mapping/000000.png
  visualization/color_pq_mapping.png
  visualization/lut_slices.png
  visualization/calibration_summary.png
  visualization/pq_ambiguities.png
  visualization/ambiguities.npz
  visualization/summary.json
```

每图对应一个六位索引缓存名、一张合并预览和一行报告；manifest 固定记录索引、完整相对 key、源路径和参考路径。失败使用稳定 reason code，并排除出训练。预览在阶段重跑时覆盖同一文件。

| 文件 | 关键内容 |
|---|---|
| 坐标图 v1 | UUID、原生轴/范围、累计弧长、行列总长和 st 表，float64，只保存数组 |
| 几何 v2 | 三套 XYZ/UV/coordinates/st/mask、16×3 系数、相机、模板、原图/区域、marker 观测与尺寸、拟合信息和 geometry_stage |
| prepared | `geometry_stage=preview_reference`，可从观测重算；不允许直接训练 |
| overall | `geometry_stage=final`，含 `marker_fit_valid` 和最终系数 |
| frames | 最终 XYZ/UV/st/原生参数、参考/当前颜色和有符号差、几何/光度/sample mask、球面坡度/法向/球心/根分支与诊断 |
| 样本 v1 | 逐点颜色、坡度、图片分组、st/原生参数、局部法向与切向基，参考背景和几何模型 |
| LUT v5 signed | 坡度、方差、原始节点计数/有效性、颜色范围及 Unicode JSON metadata |
| 冻结采样网格 v1 | 每次图片 fit 通过后从同批 reference 重建；规则 st、float64 原生材料坐标、坐标编号、相机、模板和参考来源 |

`xyz / uv / coordinates / st / valid_mask / geometry_valid_mask` 是 observation 网格别名。
较低分辨率 geometry 网格使用 `geometry_grid_valid_mask`，避免覆盖 observation 的 mask；缓存加载兼容旧字段。
`*_coordinates` 始终为原生 a,b，不保存归一化 st 到该字段。

全部 NPZ 使用 `allow_pickle=False`，不保存 object 数组。系数、相机、坐标图和冻结逆采样网格用 float64；普通几何采样网格、颜色和坡度用 float32；mask 用 bool；字典用 Unicode JSON。缓存通过模板和系数恢复曲面，不重新拟合缓存 XYZ。

所有新几何文件声明 `coordinate_kind=reference_arclength`、`coordinate_chart_id`、相对当前 NPZ 的 `coordinate_chart_file`、`frame_convention=st_grid_columns_rows`。颜色/样本还声明 `residual_domain=material_grid` 和 `color_space=linear_rgb`。

## 6. 局部推理

`recon-local` 先读取 LUT 元数据，再检查输入的坐标身份、基约定和颜色域。可以直接使用 `frames/*.npz` 中的最终几何与曲面色差：

```bash
uv run recon-local --input assets/normal_calibration/frames/000000.npz \
  --calibration assets/normal_calibration/normal_lut.npz --output local.npz
```

自备 NPZ 可传 `color_residual_grid`，或 `observed_color_grid / reference_color_grid`，并提供 `photometric_valid_mask`。原图模式必须提供各自的 `uv / reference_uv`、相同材料身份的 `st / reference_st` 和 `material_coordinates / reference_material_coordinates`，以及两侧 surface/marker mask（或显式 source_valid_mask）。原图各自采样后再相减。新坐标路径拒绝全图预先相减的 `residual_linear_rgb`。

局部位移的基准是输入包的 `xyz`。普通 `reconstruction/get_surface.py` 流程中，它来自当前形变帧的 marker 整体拟合。求解使用该曲面的法向和曲率，输出为 `xyz_out = xyz + displacement * normal`；位移单位为 mm。参考图的整体重建用于冻结材料坐标和配对参考颜色，此处不通过参考 XYZ 相减来计算位移。因此，若当前整体拟合已吸收一部分按压形变，局部位移表示对该拟合曲面的修正，不能直接当作相对未按压曲面的总位移。

`local_reconstruction.zero_color_protection` 在查询 LUT 后分别处理正、负色差。
`inner_radius / outer_radius` 控制正方向，`negative_inner_radius / negative_outer_radius` 控制负方向，
四个参数均填写非负幅值；省略负方向参数时分别继承正方向值，兼容旧配置。

令正向幅值 `P=max(0, max(dRGB))`、负向幅值 `N=max(0, max(-dRGB))`。
两侧分别计算 `t=clip((幅值-inner)/(outer-inner), 0, 1)` 和 `h(t)=t²(3-2t)`，
取两侧响应的较大值 `A`，将 LUT 的两个坡度一起乘以 `A`，保留坡度方向。
这样混合正负 RGB 中的有效变化不会被另一侧较大的原始幅值掩盖。

当前正方向阈值为 `0.08 / 0.15`，负方向阈值为 `0.03 / 0.08`：
负色差从 `0` 到 `-0.03` 位于归零区，从 `-0.03` 到 `-0.08` 平滑恢复，
达到 `-0.08` 及更负时使用完整坡度。整个 RGB 的两侧都在归零区内才按无接触先验处理，
有效颜色点的坡度归零、置信度设为 `1`；过渡区及完整响应区保留 LUT 置信度。
marker、饱和或其他光度无效点保持置信度 `0`。

局部重建的颜色方程等权参与最小二乘，仅按几何与光度 mask 决定是否入解，不按置信度筛选，也不乘置信度或面积权重。置信度仅保存到 NPZ 作诊断，不参与求解或预览绘制；改变置信度不会改变位移。孔洞仍按物理距离建立连续性方程，显式可信无接触点的软零位移约束保留 `sqrt(lambda)` 系数；两者均不使用置信度。

同一方向的两个半径均为 `0` 时，该方向的非零分量不应用保护；四个参数均为 `0` 时关闭整层保护。
只要仍有方向启用保护，零色差仍按零坡度先验处理。
调整阈值只需重新运行重建，无需重新标定；不会直接把这些点的位移截成零。
原始 LUT 方差保留为独立诊断量，置信度 `1` 表示施加了零坡度先验。
LUT 构建时的零色差锚点，以及显式 `trusted_no_contact_mask` 对坡度和位移的约束，是独立机制。

未知位移覆盖几何内部；颜色方程只覆盖光度有效点。marker 孔洞通过几何四邻域加权连续性连接，不作为零位移边界；仅真正外边界沿用零位移条件。缺失区输出法向从重建 XYZ 推断，另存 `inferred_mask`，光度 confidence 仍为零。没有有效内部颜色方程时拒绝求解。

已有重建结果可直接显示，不需要重新加载输入包、配置或 LUT：

```bash
uv run local-recon --view assets/reconstruction/pc/000000_local_result.npz
```

窗口仅显示两张图：相机坐标系中的三维曲面、局部法向位移图，不绘制置信度图。三维曲面按局部位移着色，XYZ 与位移单位均为 mm；正位移沿整体基准法向。三维坐标轴保持相同物理尺度，可用鼠标拖动旋转。二维图的行列是材料采样网格索引，灰色表示无效几何，marker 处仍显示求解出的位移。预览只读取 `xyz_out / displacement / valid_mask`，不要求 NPZ 含置信度字段。

二维位移图的显示框按曲面实际宽高比设置，不按网格列数与行数决定比例。
分别累加行、列相邻有效 XYZ 的三维距离，并取各条曲线长度的中位数作为代表宽度与高度；
缺失段按同条曲线的可见平均步长估算。该比例用于整个显示框，不表示弯曲曲面能严格无失真地展开到平面。

完整重建和独立局部重建默认将 NPZ 保存到 `reconstruction.output_dir/pc/`，同名 PNG 直接保存到 `reconstruction.output_dir/`；当前分别为 `assets/reconstruction/pc/` 和 `assets/reconstruction/`，目录自动创建。显式 `--output` 按给定位置保存 NPZ，默认预览图放在 `pc/` 上一级目录，其他位置则与 NPZ 相邻。加 `--show` 可在保存后显示窗口。`--preview-output` 可自定义预览路径或改存 JPG/SVG/PDF；与 `--view` 配合且不加 `--show` 时只保存图片，适用于无图形界面的环境：

```bash
uv run local-recon --view assets/reconstruction/pc/000000_local_result.npz \
  --preview-output assets/reconstruction/000000_local_result.png
```

旧 v1 几何和 v4/v5 LUT 可按旧模板参数路径读取；不会自动转换成新的参考弧长语义。本次修改 CPU 路径，JAX 独立数值接口保留。

## 7. 单张图片完整重建

整体重建、局部重建和预览均位于 `reconstruction/`，`reconstruction/recon.py` 为完整流程入口，命令为 `uv run recon`。
原来的局部重建位于 `reconstruction/local_recon.py`，原 `recon-local` 命令仍可使用，也可使用 `local-recon`。

```bash
uv run recon --config config.yaml \
  --image assets/cali_norm_pics/capture_20261016_184356_361449.png
# 可指定结果路径并显示预览：
uv run recon --image input.png --output assets/reconstruction/pc/result.npz --show
```

配置段 `reconstruction` 指定 `reference_image`、`material_grid_file`、`output_dir` 和 `match_surface_aspect`。
程序用参考图固定 marker ID，独立拟合当前曲面，按与 LUT 配套的材料网格分别采样两张图，
然后调用相同的局部求解和保存函数。默认在 `output_dir/pc/` 保存 `<图片名>_local_result.npz`，同名 PNG 保存在 `output_dir/`，不生成中间输入包。
相机、模板、材料身份及颜色约定不一致时拒绝使用；不自动训练或替换 LUT。

`assets/normal_calibration/lut_material_grid.npz` 随图片法向标定的最终 LUT 更新，
每次从同批 reference 坐标图生成，基准尺寸取本批 reference 的 observation 网格。
默认 `match_surface_aspect: true`：按当前整体曲面三维曲线的代表宽高确定求解网格，
长边点数取 observation 配置的较大维度，短边间隔数取长边间隔数乘物理短长比后四舍五入。
冻结逆网格在共同规则 st 上插值原生参数，参考与当前 XYZ、UV、颜色均重新计算；重建期间 LUT 与网格文件不变。
也支持匹配 LUT 的完整参考弧长图，直接反解新网格。
设为 `false` 时严格使用 observation 行列数。独立 `local-recon --input` 仍使用输入包自带网格。
旧批次的原始 17 张标定照片及参考图保留在 `lut_source_images/`，来源记录不作为当前 LUT 的配套声明。
过期光场素材、模型、实验、测试重建结果和未使用的训练检查点已清理；
当前法向标定资料、新批次标定照片与相机标定保留。
完整配置和命令见 [重建模块说明](../reconstruction/README.md)。

## 8. 模块职责

| 位置 | 职责 |
|---|---|
| `reconstruction/recon.py / reconstruction/pipeline.py` | 单张图片的整体曲面、颜色配对和局部重建编排 |
| `calibrate/calibrate_camera.py` | 相机标定，保留原入口 |
| `calibrate/calibrate_norm.py` | 参数解析和阶段选择，重导出 build_normal_lut |
| `calibrate/norm_pipeline.py` | 图片准备、缓存、重拟合、颜色/标签组合、样本和整图验证 |
| `calibrate/visualize_norm.py` | 标定后的色差、p/q 映射、LUT 切片及拟合/独立验证可视化；支持独立补生成 |
| `calibrate/manual_norm_regions.py` | 接触语义和三面板回调适配 |
| `reconstruction/surface_reconstruction/markers.py` | 黑框区域、marker 检测与光度屏蔽 |
| `reconstruction/surface_reconstruction/tracking.py` | 完整网格初始化与固定 ID 跟踪 |
| `reconstruction/surface_reconstruction/surface.py` | 双三次拟合与 BicubicSurfaceAdapter |
| `reconstruction/surface_reconstruction/run_reconstruction.py` | session、provider 和独立调试入口 |
| `utils/surface_geometry.py` | 参数曲面协议、共用局部几何、法向/坡度转换 |
| `utils/surface_sampling.py` | 精度保持的四邻域采样和解析 Jacobian |
| `utils/surface_coordinates.py` | 弧长积分、冻结坐标图、正反解与多分辨率缓存 |
| `utils/surface_projection.py` | 相机、投影、射线与唯一曲面逆投影 |
| `utils/sphere_geometry.py` | 固定半径球拟合、双根求交和球冠选择 |
| `utils/region_annotations.py` | 通用区域数据、YAML、边界和 mask |
| `utils/annotation_editor.py` | 仅依赖回调的通用编辑器 |
| `utils/annotation_window.py` | 固定图像范围的标注窗口，屏蔽导航缩放并转换鼠标与键盘事件 |
| `utils/color_difference.py` | 线性 RGB、有符号差和显示转换 |
| `utils/normal_mapping.py` | LUT 数据、构建、原始查询 |
| `utils/reconstruction_interface.py` | 三套网格和坐标类型 |
| `reconstruction/get_surface.py / reconstruction/local_recon.py` | 文件适配、CLI 与推理策略；兼容导出旧接口 |
| `reconstruction/viewer.py` | 局部重建 NPZ 的三维曲面、位移两图显示和图片导出；由保存函数及 `local-recon --view` 调用 |
| `slope_color_audit.py` | 独立离线诊断：对照色差和坡度，统计位移正负交界，输出图、CSV、JSON；不参与主流程 |

底层 utils 不反向导入 calibrate，不读配置、不创建 GUI、不读取 provider 的隐式状态。采集工具保留，曲面有效域由黑框内部轮廓提供。仓库不包含背景光场生成算法。

`slope_color_audit.py` 要求旧式成套输入：原图、整体几何 NPZ、`*_recon_input.npz` 和 `*_local_result.npz`，并使用能复现结果的 LUT 与保护参数。
当前单图入口只保存最终结果及预览，不能仅凭它们执行完整审计；此工具保留用于已有输入包的分析。

## 9. 开发验收

```bash
.venv/bin/python -m unittest discover -s tests -p test_norm_calibration_pipeline.py -v
```

标准库 unittest 覆盖：弧长/坐标闭环及多解拒绝；独立颜色采样和几何缺失区；两侧球面几何及法向符号；无界面样本/拟合、缓存复用、分组验证隔离和失败不覆盖发布；单图流程与离线求解一致、参考图零位移和材料身份校验。实际数据运行后查看每图合并预览、`report.csv` 与 `validation.json`。

`assets/`、模型、训练数据、缓存和输出为本地资产，不纳入 Git。验收模块 `tests/test_norm_calibration_pipeline.py` 单独解除测试目录忽略，便于版本管理。
