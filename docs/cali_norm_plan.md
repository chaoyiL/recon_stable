# 压球图片的色差—法向标定实施计划

本文记录压球图片标定的处理链路、算法与接口。2026-11-06 已将现有标定程序改为先重建当前整体曲面，再以接触位置和法向压入量驱动球—曲面交线预览；第 1.3、4、5 节及相关接口、文件契约已同步。历史设计中其余“新增”“调整”的实际状态以代码及 [structure.md](structure.md) 为准。

## 核心功能与完整链路

标定需要完成三件事：重建胶皮整体曲面并建立公共曲面坐标；在公共坐标上取得色差和球面法向；将成对样本训练成 LUT，并用整张留出图片验证。

```text
固定底色图 → 黑框内部区域、marker → 参考整体曲面 Q_ref
                                         ↓
                                 参考弧长坐标图 F_ref
                                         ↓ 全批次共用
压球图片 → 黑框内部区域、marker 跟踪 → 当前整体曲面及曲率
                                         ↓
                                拖动接触位置、滚轮调压入量
                                         ↓ 排除接触区 marker
                                   最终整体曲面 Q_i
                                         ↓
                   同一规则 s,t 网格 → 原生参数 a,b → XYZ、UV
                              ↙                         ↘
          分别采样压球图和固定底色图              接触位置/压入量 + Q_i + 球半径
                     ↓                                  ↓
           曲面域线性 RGB 有符号差                 球心、球冠、法向
                              ↘                         ↙
                             色差—局部坡度成对样本
                                         ↓
                                LUT → 整图留出验证
```

**色差只在曲面域计算。** 当前图与底色图分别经各自的 UV 采样到同一个材料网格，再相减。黑色 marker 参与几何重建，但不提供颜色样本。

当前代码已提供 marker 检测与跟踪、黑框区域 mask、双三次整体曲面、参考弧长坐标、接触排除重拟合、位置/深度交互、球面法向标签、图片批处理与 LUT 验证。以下保留链路说明；历史实施项的现状以实际接口为准。

以下第 1—7 节说明功能，第 8 节确定代码结构，最后给出阶段、文件、校验和实施顺序。

## 1. 从图片重建整体曲面

### 1.1 复用 marker 检测得到胶皮区域

当前 `reconstruction/surface_reconstruction/markers.py:find_inner_region` 已输出内部轮廓、填充轮廓得到的 `surface_mask` 和 marker 搜索区域；兼容的 `find_inner_roi` 仍返回矩形 ROI。

实现要求：

- `surface_mask` 是黑框内部的完整胶皮区域，包含 marker 所在位置；不要用矩形填充代替实际轮廓。
- marker 搜索使用基于黑框距离和实际十字尺寸的独立 `marker_mask`，不按固定图像横坐标裁剪十字。胶皮轮廓和 `surface_mask` 保持完整；检测坐标仍为原图坐标。
- `MarkerTracker.update` 的 ROI 是随跟踪点移动的搜索窗口，不能作为胶皮 mask。
- 参考图和每张压球图分别检测黑框。检测失败时记录原因并跳过该图。
- 旧 `find_inner_roi(gray)` 保持返回矩形；新增函数返回完整区域对象，由旧函数包装调用。

将这一结果贯通 session、provider 和 `reconstruction/get_surface.py`。图片标定和 `get-surface` 都从 marker 检测取得胶皮区域，不再初始化独立分割后端。其他训练工具不属于本次修改范围。

### 1.2 参考图：建立固定材料身份

1. 加载相机内参、畸变、物理 marker 模板和网格配置。
2. 在底色图中检测黑框和完整 marker 网格，固定物理 ID 与横纵方向。
3. 拟合参考整体曲面 `Q_ref(a,b)`，通过现有整体重建门槛。
4. 按第 2 节建立并保存唯一参考弧长图。
5. 保存参考系数、相机、模板、marker 位置与尺寸、胶皮 mask 和坐标网格。

当前模板为 48 个 marker；参考初始化仍要求完整网格。模板存在方向对称性，参考图确定方向后必须保存并复用，不能对每张图自由翻转或重编号。

### 1.3 压球图：先整体重建，再交互和最终拟合（已实现）

每张独立照片都从同一参考图初始化 ID，绑定同一参考弧长图。跟踪当前图后，用参考系数作先验拟合当前整体曲面，缓存为 `geometry_stage=initial`。参考曲面不再直接充当当前图预览；旧 `preview_reference` 缓存在打开标注或生成样本时先从观测重新拟合。

当前整体曲面提供曲率和初始球接触预览。按 `g` 时用生成的接触轮廓外扩 20 px 排除 marker，重新拟合后再按相同图像锚点与深度生成轮廓。重复至输入拟合的 marker 排除集合稳定，最多 6 次；不收敛则拒绝。每次都使用同一参考先验，最终颜色、球心、轮廓和标签来自同一最终曲面。

排除接触点只改变拟合输入，不删除跟踪位置；颜色屏蔽仍需要这些位置。重复标注复用跟踪缓存，不推进光流。只有 `final` 几何可供训练。

## 2. 从整体曲面建立可复用的 s,t 坐标

### 2.1 三类坐标必须分开

| 坐标 | 含义 | 单位 |
|---|---|---|
| `a,b / parameters` | marker 模板定义的原生材料参数；当前是模板 x、y | mm |
| `s,t / st` | 参考曲面上两族材料曲线的归一化累计弧长 | 无量纲 |
| `u,v / uv` | 带相机畸变的原图坐标 | pixel |

已有 provider 能从原生参数计算 XYZ 和 UV，尚未实现弧长积分及其逆映射。

沿参考曲面的两族曲线 `a→Q_ref(a,b_const)`、`b→Q_ref(a_const,b)` 定义：

```text
ell_a(a,b) = integral[a_min..a] ||∂Q_ref/∂a(alpha,b)|| d alpha
ell_b(a,b) = integral[b_min..b] ||∂Q_ref/∂b(a,beta)|| d beta

s(a,b) = ell_a(a,b) / ell_a(a_max,b)
t(a,b) = ell_b(a,b) / ell_b(a,b_max)
F_ref(a,b) = (s,t)
```

积分使用三维速度范数，不能使用某个 XYZ 分量、UV 距离或模板坐标的 min/max 缩放。例如 `Q=(a,b,a²)`、`a∈[0,1]`，在 `a=0.5` 时 `s≈0.388045`。

这两个标量场通常相互耦合，既不保证正交，也不保证等距。物理距离和面积仍由 XYZ 及其度量计算；不能把归一化 st 距离当成毫米距离。[曲面弧长与第一基本形式](https://kodu.ut.ee/~abramov/teaching/DifferentialGeometry/Monograph_files/Part1_Main.pdf#page=87)

### 2.2 构建参考弧长图

在完整原生参数矩形上建图，不按可见 mask、marker 孔洞或输出分辨率裁剪：

1. 从 `129×129` 原生规则网格开始，计算参考曲面一阶导数。
2. 双三次曲面的导数使用 `model.basis(parameters, dx=1, dy=0) @ coefficients`；另一方向改为 `dx=0, dy=1`。
3. 对导数范数沿对应原生轴累计积分，再分别除以每条曲线的总长。
4. 加密为 257、513、1025；共享节点的 st 最大绝对差与行列总长最大相对差均 ≤`1e-5` 才接受较细结果，达到上限仍不满足则报错。
5. 保存原生轴、累计长度、行列总长和 st 表；几何计算及插值均使用 float64。

```python
# 网格形状 Nb, Na；a 为列，b 为行
ell_a = cumulative_trapezoid(speed_a, x=a_axis, axis=1, initial=0)
ell_b = cumulative_trapezoid(speed_b, x=b_axis, axis=0, initial=0)
row_lengths = ell_a[:, -1]
column_lengths = ell_b[-1, :]
st_table = np.stack(
    [ell_a / row_lengths[:, None], ell_b / column_lengths[None, :]],
    axis=-1,
)
```

积分接口使用 [SciPy cumulative_trapezoid](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_trapezoid.html)。速度、曲面度量或长度退化时拒绝建图。

对表格的每个双线性 cell 检查四角 Jacobian。在归一化原生参数 `alpha,beta∈[0,1]` 下要求 `det d(st)/d(alpha,beta)>1e-8`，并检查四条边分别对应 s/t 的 0/1 边界。失败报告坐标图不可逆，不退回线性缩放。本次只支持通过这些条件的单张参数片。

### 2.3 反解、重采样与跨图片复用

正向映射使用同一份 st 表的双线性插值；逆映射是二维耦合求解：

1. 建立 st 表节点的 KDTree，最近节点提供原生参数初值。
2. 在归一化原生变量 alpha、beta 上做 Newton，使用同一双线性表达式的解析 Jacobian。
3. 最多 16 次，每步最多 8 次减半回溯，要求不出参数域且残差下降。
4. st 残差无穷范数 ≤`1e-8` 才接受；任意点查询返回逐点有效性。
5. 规则输出网格全部位于支持域，任一点反解失败都应报错，不能算作图像不可见。

表格索引导数转换到 alpha、beta 时，右乘 `diag(Na-1,Nb-1)`；转换到原生 a、b 时，还需分别除以原生轴跨度。端点使用最后一个完整 cell，权重允许为 1。

随后真正重采样曲面：

```text
规则目标网格 st
    → F_ref^-1(st) 得到原生 parameters
    → Q_i(parameters) 得到当前 XYZ
    → 相机投影得到当前 UV_i

同一 parameters → Q_ref(parameters) → 底色图 UV_ref
```

不能只给旧的原生均匀网格加一个 st 标签，也不能把 st 直接传给双三次 basis。双三次系数始终表示 `Q(a,b)`。

参考图建立 `F_ref` 后，全批次冻结复用。同一 st 始终代表同一原生材料身份；当前曲面的弧长变化可以另作诊断，不能重新归一化后与参考图按数值相同的 st 配对。

三个输出分辨率共用同一坐标图，反解按 `(chart_id, rows, columns)` 缓存。图的 UUID 在保存、加载和回放中不变。

普通 `get-surface` 增加 `--coordinate-reference PATH` 以加载已有图；未指定时由首个有效曲面建图并保存。provider 重置跟踪默认保留坐标图，只有显式要求才清除。

## 3. 在同一曲面网格上采样颜色

### 3.1 固定底色图替代背景光场

本次用固定底色图采样替代背景光场生成。顺序固定如下：

```text
底色线性 RGB → 按 UV_ref 采样 → C_ref(s,t)
压球线性 RGB → 按 UV_i   采样 → C_i(s,t)
delta(s,t) = C_i(s,t) - C_ref(s,t)
```

保留有符号的三个通道，不取绝对值、不转灰度、不做显示归一化。固定底色假设由最终曲面上的非接触区色差验证，不因“形变近似为零”省掉曲面坐标转换。

已实现：相机帧和旧编码图片只在入口按 BT.709 逆曲线线性化；新图片保存浮点 NPZ，读取后直接使用。标定与重建共用 `utils/image_io.py`，缓存持有线性整图，样本生成和重算不重复解码。所有原图预览直接显示线性值，不做显示曲线转换，也不自动保存检查 PNG。曲线、当前设备核对结果及旧样本/LUT 的重算要求见
[相机颜色契约](structure.md#相机颜色契约2026-11-02-核对)。

参考、当前网格的 chart_id、st、原生 parameters 必须一致；各自使用自己的 XYZ、UV 和有效域。

### 3.2 黑色 marker 只造成光度缺失

每张原图先建立：

```text
source_valid = surface_mask & ~marker_image_mask & unsaturated
```

`unsaturated` 要求线性 RGB 三个通道均小于 1，对应原始 BGR8 的 255 饱和阈值。marker mask 由固定 ID 的位置和局部黑色连通域取得，包含组件形状及边缘保护带；参考图和当前图分别建立。

复用现有组件检测，但保持 `detect_cross_centers` 的三个返回值兼容。提取公共的组件分析辅助函数，避免再写一份阈值和连通域算法。组件对应不唯一、暂时失检时，按参考尺寸和已跟踪位置保守屏蔽。搜索半径与边缘扩张值见第 12 节。

颜色采样采用严格四邻域双线性插值：四个源像素都在图像内、source_valid 且有限才有效。图像最后一行/列不具备完整邻域，不能复制边界凑齐。两张图分别采样后，将两侧有效性与几何有效域相交，得到 `photometric_valid_mask`。

不对 marker 黑色区域插值补色后训练；也不把它当作零色差、零法向变化或零位移。几何 mask 始终保留这些位置。

## 4. 可视化标注接触区域（已实现）

### 4.1 位置、深度与实时轮廓

三面板依次显示原图、投影回原图的曲面色差、整体主曲率预览或重算后的球面法向。宽度为 640、960、280 px，图像区高 720 px；鼠标位置扣除中间面板偏移、除以显示比例后转换为含畸变原图像素。

拖动绿色轮廓任意位置平移接触锚点，滚轮每格改变 `0.01R` mm 的法向压入量。每次移动或深度变化都立即求解第 5 节的交线；轮廓由三维曲面和投影确定，可以不是圆或椭圆。保留 `c` 设置锚点、`z` 撤回、`g` 重算、`Enter` 验证并接受、`x` 跳过及 `Esc` 保存草稿退出；不提供手调半径、旋转或顶点编辑。

深度约束为 `0<d≤R`，允许达到 `d=R`。零深度接触退化为点，不能用于面域标定；编辑下限为 `1e-4R`。面板显示当前和最大压入量，橙线显示当前锚点处 `d=R` 的投影轮廓；最大轮廓超出可见曲面时不画橙线，不影响更浅的有效接触。无效拖动恢复上次有效状态并显示原因。重算导致轮廓变化时清除过期结果，须再次重算才可接受。

标注语义保持 `contact_boundary`，`center_px` 现在明确表示压入轴与未局部变形的整体曲面的交点投影，不是三维球心投影。普通 ROI 或可见球外轮廓不能当作接触边界。

### 4.2 标注、mask 与旧数据

标注使用 `shape=sphere_contact`、`indentation_depth_mm` 和求解得到的 `contour_px`，位置与深度是重算输入，保存轮廓用于回放。接触轮廓填充为标签区域，不内缩；外扩 20 px 仅用于整体拟合与背景统计排除。

旧圆、椭圆及多边形 YAML 仍能读取。打开旧接触标注时保留中心，用其三维边界尺度估计初始深度，再由新模型生成轮廓并要求人工确认；不会继续走旧的自由球心拟合。已有新格式 accepted/skip 默认复用，`--edit` 可重开。文件格式见第 10.3 节。

## 5. 从位置、深度和整体曲面计算球面法向（已实现）

### 5.1 几何假设与球心

沿用 `boundary_model=zero_boundary_displacement`：接触边界同时位于未局部变形的整体曲面与球面上。真实接触还可能受材料弹性、摩擦等影响，本模型不求解这些力学作用。所有 XYZ、球心及射线都在相机坐标系，长度单位 mm；参考法向指当前整体接触曲面的单位法向。

以含畸变原图像素 `center_px` 在当前整体曲面 `Q(a,b)` 唯一逆投影，取得 `A=Q(a0,b0)`。无解或多解拒绝；逆投影及误差尺度使用共用 `invert_surface_projection`。由曲面一阶导数取得 `e1=normalize(Q_a)`、`n0=normalize(Q_a×Q_b)`、`e2=n0×e1`。

球真实半径为 `R>0`，法向压入量为 `d`，球心侧 `sigma=+1/-1` 对应 `positive_normal/negative_normal`：

```text
C = A + sigma*(R-d)*n0
球的轴向压入端点 = A - sigma*d*n0
0 < d <= R
```

由用户指定位置和深度直接构造球，不再从轮廓拟合自由球心。`d=R` 时 `C=A`，侧别仍由配置保留，用来选择法向和球冠。

### 5.2 主曲率

对规则 st 网格的三维点，用共用 `compute_surface_geometry` 求第一基本形式 `G=JᵀJ`、第二基本形式 `B_ij=n·Q_ij`。设局部正交切基矩阵为 `E=[e1,e2]`，则 `K=EᵀJG⁻¹`，对称矩阵 `K B Kᵀ` 的两个特征值是带符号主曲率，单位 `mm⁻¹`。曲率与既有几何字段共用有效 mask；边缘或退化邻域无有效曲率。显示 `max(|k1|,|k2|)`，最终逐图缓存保存两主曲率。

### 5.3 直接求球—曲面交线

局部曲率不能单独确定任意有限接触区。程序直接求解完整重建曲面的隐式交线：

```text
F(a,b) = ||Q(a,b)-C||²-R² = 0
boundary_px = project(Q(a,b), camera)
```

从中心在原生参数域发出 128 个方向，每条射线划分 96 个区间，向量化采样寻找变号区间并二分细化根。参数方向只用于数值搜索，不当作三维弧长。只接受环绕锚点的闭合、横截且能投影的单接触轮廓；曲面边界截断、采样检出的多次交叉、投影退化等拒绝。有限方向和径向采样不是任意复杂曲面的拓扑证明；复杂非星形接触区不在支持范围。

投影使用真实相机内参和畸变。平面时空间接触半径精确退化为 `sqrt(2Rd-d²)`；倾斜、弯曲和透视下使用实际投影，移除旧的像素半径上限。

### 5.4 从观察射线取得球面点和法向

每个颜色样本沿其 **同一个 UV_i** 发射相机射线。去畸变后，令 `r=normalize([x_n,y_n,1])`：

```text
b = r·C
disc = b² - (C·C - R²)
t_minus = b - sqrt(disc)
t_plus  = b + sqrt(disc)
P = t*r
n_gt = -sigma*(P-C)/R
```

只接受正 t 和有效判别式。不能用整体曲面点 `Q_i-C` 代替真实球面交点 `P-C`。

先在接触中心射线上选根：候选须满足 `n_gt·n0_center>0.1` 且 `sigma*(P-A)·n0_center<=center_sigma_mm`。在合格候选中选离 A 最近者；距离差不超过 center_sigma_mm 时视为分支不确定。

将选中的 minus/plus 分支锁定到整张图，不逐点切根或翻转法向。每点还须位于最终接触轮廓内，不缩小半径或内缩边界，并满足 `n_gt·n0_center>0.1`、`n_gt·n0(point)>0.1`。

最后按最终规则 st 网格的共用局部基转换为坡度：

```text
p = -(n_gt·e1)/(n_gt·n0)
q = -(n_gt·e2)/(n_gt·n0)
n_gt = normalize(n0-p*e1-q*e2)
```

局部基必须复用现有 `recon._surface_geometry` 的行列方向与符号。法向↔坡度只实现一次，标定、验证和推理共同调用。这套标签方法不额外恢复局部面内材料运动。

## 6. 配对样本、LUT 与整图验证

每图只建立一个最终索引：

```text
sample_valid =
    geometry_valid
    & photometric_valid
    & contact_mask
    & sphere_label_valid
```

用同一个扁平 bool 索引提取 `N×3` 色差、`N×2` 坡度、st、原生参数和 image_index，再拼接。不能分别过滤颜色与标签。

复用现有 `build_normal_lut` 的节点计数、MAD、方差、缺节点补全和零点锚定。映射输出保存为局部坡度；需要法向时用共用转换函数还原。

已实现：取消 MAD 前后的最小样本数门槛，并移除对应参数和配置。MAD 保留非空样本后，仅按节点内部法向夹角 RMS 检查一致性；单样本及少于 5 个样本的节点均可参与建表。

按图片分组留出，而不是随机拆散像素：

1. 对有有效样本的图片 key 排序，用随机种子 42 打乱；至少需要两张图。
2. 留出 `min(M-1,max(1,ceil(0.2*M)))` 张，其余训练。
3. 训练颜色范围、节点及补全只使用训练图。
4. 验证调用原始 LUT 插值，不应用推理端的零色差衰减或无接触覆盖。
5. 由坡度还原局部法向，要求每张验证图角度 RMS ≤5°；同时报告总体中位数、P95。
6. 原始验证色差中，任一通道超出训练颜色范围的样本比例 ≤0.05。即使查询会 clip，也按 clip 前数据统计，角度误差包含这些样本。

先保存训练 LUT 与验证报告。通过后用全部 accepted 图片重建最终 LUT 并原子保存；失败不覆盖已有最终输出。只有旧两数组的外部样本模式可以保留，但必须标为未做整图验证，不能冒充本流程产物。

已实现配套采样网格更新：图片 `fit` 每次验证通过后，读取同批 `reference_surface.npz` 及其参考弧长图，校验参考图和坐标编号与待发布 LUT 一致。以 reference 的 observation 行列数生成规则 st，调用同一 `sample_surface_grid` 反解原生材料参数，写入 `work_dir/lut_material_grid.npz`；不读取旧网格，不继承旧分辨率。两个临时文件写完后再分别替换最终 LUT 与网格。验证、reference 检查或临时写入失败时保留已发布产物；旧两数组样本不生成网格。

保留显式科学分析工具，正常标定不自动调用：`calibrate/visualize_norm.py` 是命令入口，调用 `calibrate/utils/visualize_norm.py`，使用本批 LUT、聚合样本及 `frames/overall` 缓存，在 `work_dir/visualization/` 生成逐图色差与 p/q 对照、RGB 色差到 p/q 的三维映射、三个零通道 LUT 切片和整批汇总。逐图预测调用最终 LUT 的原始插值；独立留出统计调用同批 `_train.npz`，复用已有局部坡度转法向函数计算角误差。显示回投复用 `rasterize_surface_grid`，统计始终使用原材料网格样本。所有输入校验坐标身份；空间样本还检查与训练样本逐点一致，不混入遗留缓存。显式分析失败不改写已发布 LUT 的验证状态；完整报告生成后才替换旧报告。没有图片分组的旧两数组模式不生成该报告。

已实现 p/q 不良映射图：使用拟合及剔除前的全部配对样本，以 KDTree 查询有符号线性 RGB 欧氏距离 ≤0.01 的所有无序样本对，包含同图与跨图。将 p/q 转成公共局部基下的单位法向 `normalize([-p,-q,1])`，夹角 >20° 的样本对为冲突对，两端点均在 p/q 图中标红。一个点可参与多个冲突对，点数与对数分别统计；不只选最近邻。两个阈值可通过可视化 CLI 调整。图片仅输出 PNG，另保存全部冲突索引、逐对距离与角度、不良点 mask 的 NPZ，汇总 JSON 记录阈值和统计。这些诊断不改变 LUT 拟合或筛选规则。

已实现失败模型诊断：独立绘图入口接受包含失败验证报告的 `validation_status=failed` 诊断 LUT，所有图表明确标记失败且未发布。汇总同时记录验证状态、验证报告、有效节点数及保留样本数；留出统计仍使用配套 `_train.npz`。这不会放宽验证或发布门槛。

## 7. 推理端复用同一约定

`recon-local` 同步调整，否则训练与使用时的色差含义会不同：

- 原图输入也必须各自采样后相减；允许直接读取已在曲面域计算的颜色或残差，但须匹配坐标图、局部基和颜色域声明。
- 先读取 LUT 的元数据，再校验输入的 `coordinate_kind / coordinate_chart_id / frame_convention`。
- 分开 `geometry_valid_mask` 与 `photometric_valid_mask`。位移未知量覆盖几何内部，颜色方程仅覆盖光度有效点。
- marker 孔洞保留为未知位移，通过邻域连续性约束连接；只有真实外边界施加原有零位移条件。
- 启用零色差保护时，正向幅值 `max(0,max(dRGB))` 与负向幅值 `max(0,max(-dRGB))` 分别按独立的归零/恢复阈值计算 smoothstep 响应，取较大响应统一缩放两个坡度。两侧响应均为零的光度有效点按无接触先验处理，坡度为零、记录的置信度为 1；其余点保留 LUT 置信度。参数与关闭保护的约定见 [结构说明](structure.md)。这项颜色判定不直接增加绝对零位移约束。
- 颜色方程等权求解，不乘置信度或面积权重，也不按置信度筛选方程。置信度仅供保存诊断，不进入求解或预览；几何及光度有效性仍由各自的 mask 决定。局部重建预览只显示三维曲面和法向位移两张图。
- 缺失区输出法向可由重建后的 XYZ 求得，并标为 `inferred`，不能提高其光度置信度。完全没有有效颜色方程时应报错。

缺失点 i 使用已有几何四邻域的加权连续性：

```text
w_ij = 1 / max(||Q_j-Q_i||², 1e-12)
a_ij = w_ij / sum_j(w_ij)
h_i = sqrt(mean_j(||Q_j-Q_i||²))
(d_i - sum_j(a_ij*d_j)) / h_i = 0
```

连续性方程不再额外乘面积或置信度权重。只在光度缺失内部使用，不给所有点额外加平滑项。显式可信无接触点的软零位移约束使用 `sqrt(lambda)` 系数。

## 8. 代码结构与确定接口

### 8.1 模块边界

| 位置 | 职责与改动 |
|---|---|
| `reconstruction/surface_reconstruction/markers.py` | 输出黑框区域与完整 mask；共用黑色组件分析、marker 光度 mask |
| `reconstruction/surface_reconstruction/surface.py` | 新增 BicubicSurfaceAdapter，包装现有 basis 和系数 |
| `reconstruction/surface_reconstruction/run_reconstruction.py` | 拆分跟踪/拟合，传递区域 mask，绑定公共坐标图并重采样 |
| `utils/surface_geometry.py` | 参数曲面协议；从 recon 提取局部几何；法向↔坡度 |
| `utils/surface_sampling.py` | 通用双线性场采样、有效邻域与同一表达式的 Jacobian |
| `utils/surface_coordinates.py` | 曲线长度、参考弧长图、正反映射、重采样及图文件读写 |
| `utils/surface_projection.py` | 相机模型、投影、射线与曲面逆投影 |
| `utils/sphere_geometry.py` | 射线双根、球冠选择、球外法向 |
| `utils/sphere_contact.py` | 位置/深度构造球、完整参数曲面交线和相机投影 |
| `utils/region_annotations.py` | 区域数据、读写、边界采样、mask 与边距 |
| `utils/annotation_editor.py` | 通用面板和形状编辑；通过回调重算、保存 |
| `utils/annotation_window.py` | 固定图像范围的 Matplotlib/PyQt5 窗口，滚轮只转交给形状编辑 |
| `utils/image_io.py` | 线性 NPZ 读写、旧图入口解码、直接线性显示适配与检测灰度适配 |
| `utils/color_difference.py` | BT.709 逆传递、曲面数组有符号差及色差示意 |
| `utils/normal_mapping.py` | LUT 数据、构建与原始查询 |
| `calibrate/utils/manual_norm_regions.py` | 为通用编辑器配置接触语义、三面板与重算回调 |
| `calibrate/utils/norm_pipeline.py` | 图片准备、最终重拟合、颜色/标签组合、样本导出、验证 |
| `calibrate/calibrate_norm.py` | 保留现有 CLI 入口，解析参数并选择阶段 |
| `calibrate/visualize_norm.py / calibrate/analyze_ambiguities.py` | 质检命令入口，解析参数并调用实现 |
| `calibrate/utils/visualize_norm.py / calibrate/utils/analyze_ambiguities.py` | 标定可视化、歧义分析与改进实验的实现 |
| `reconstruction/get_surface.py / reconstruction/local_recon.py` | 薄编排、文件适配和已有推理策略，不复制底层算法 |

依赖单向：底层数组与几何模块 → provider/标定适配 → CLI。仓库根目录的 `utils/` 不得反向导入 `calibrate`；`calibrate/utils/` 承载标定专用实现，不反向导入命令入口。数学函数不读取配置、GUI 或 `provider.last_frame_result`。

下列接口中的普通数组默认支持批量输入；参数曲面输入 Nx2 原生参数、输出 Nx3。无效值通过 mask 返回，不以零值代表成功。

### 8.2 曲面、坐标与采样接口

```python
class ParametricSurface(Protocol):
    parameter_bounds: tuple[float, float, float, float]  # amin,amax,bmin,bmax

    def evaluate(self, parameters: np.ndarray) -> np.ndarray: ...
    def first_derivatives(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def compute_surface_geometry(xyz, geometry_valid_mask): ...
def normals_to_local_slopes(normals, reference_normals, tangent_frames, minimum_dot=0.1): ...
def local_slopes_to_normals(slopes, reference_normals, tangent_frames): ...


def measure_coordinate_curve_lengths(surface, parameter_bounds, integration_shape): ...
def build_arclength_chart(surface, *, initial_size=129, maximum_size=1025,
                          tolerance=1e-5) -> ArcLengthChart: ...
def sample_surface_grid(surface, chart, rows, columns): ...
def save_arclength_chart(path, chart) -> Path: ...
def load_arclength_chart(path) -> ArcLengthChart: ...


def bilinear_sample(field, xy, source_valid_mask=None, *,
                    allow_boundary=False, with_jacobian=False) -> SampledField: ...
def sample_linear_rgb_to_surface(linear_rgb, uv, source_valid_mask): ...
def sample_mask_to_surface(image_mask, uv): ...
```

数据约定：

| 对象 | 确定字段与返回值 |
|---|---|
| `ArcLengthChart` | chart_id、coordinate_kind、parameter_bounds、a_axis、b_axis、st_table、arc_lengths_mm、row_lengths_mm、column_lengths_mm |
| chart 方法 | `to_st(parameters)`、`to_parameters(st)`、`jacobian(parameters)` 分别返回 `(values,valid)`；Jacobian 对原生 a,b 求导 |
| 长度测量结果 | 累计 ell_a/ell_b 和每条行列总长，不含归一化 |
| 网格采样结果 | parameters、st、xyz、coordinate_valid；不包含相机投影 |
| `SampledField` | values、valid、可选 jacobian；Jacobian 形状 `...×channels×2`，对表格索引求导 |
| 法向转换结果 | `(slopes,valid)` 或 `(normals,valid)` |

`BicubicSurfaceAdapter(model,coefficients)` 复制系数并实现协议，其他曲面可提供其他适配器。将 `recon._surface_geometry` 原样提取为共用函数，保留兼容别名；其中 `frame[..., :,0/1]` 是最终 st 网格列/行方向的正交单位切向。

通用采样器保持输入精度。弧长图使用 `allow_boundary=True`，颜色使用 False；两者共享插值公式。mask 包装要求四邻域都为真。坐标图数组只读，KDTree 和缓存不序列化为 pickle。

`MaterialCoordinateMap` 新增可选 st、coordinate_kind、chart_id。旧调用默认 `template_parameter`，新路径显式写 `reference_arclength`；不提供自动 min/max 推导 st 的属性。

### 8.3 区域检测与整体重建接口

```python
# markers.py
def find_inner_region(gray) -> InnerRegion: ...
def find_inner_roi(gray): ...  # 保留矩形返回值
def build_marker_image_mask(frame_linear_rgb, marker_pixels, marker_valid,
                            reference_marker_sizes, *, dilation_px=1): ...

# Session
def process(self, frame, *, fit=True) -> dict: ...
def fit_observations(self, tracking, *, exclusion_mask=None,
                     prior_coefficients=None) -> dict: ...

# Provider
def bind_coordinate_chart(self, chart): ...
def reset(self, *, clear_coordinate_chart=False): ...
def track_frame(self, frame_linear_rgb) -> dict: ...
def sample_coefficients(self, coefficients, specs, surface_mask=None): ...
def fit_tracked_frame(self, specs, surface_mask, fit_exclusion_mask,
                      prior_coefficients=None): ...
```

`InnerRegion` 固定包含 `roi / contour / mask`，其中 roi 为 marker 搜索框，mask 为原图大小 bool。session 输出中保留该对象，provider 向下传递其 mask；不依赖调用者重新检测同一帧。

`fit_observations` 只读取给定跟踪观测并拟合，不推进 tracker。`fit_tracked_frame` 对最近缓存观测调用它并返回三套坐标网格。`sample_coefficients` 直接评估已有系数，供参考预览和缓存回放使用。

普通 `reconstruct` 保持现有入口；未显式传 mask 时使用当帧 InnerRegion.mask。显式 mask 只用于进一步取交集。

### 8.4 投影、球几何与标注接口

```python
def project_points(xyz, camera): ...             # uv, valid
def unproject_rays(pixels, camera): ...          # rays, valid
def invert_surface_projection(pixels, surface, camera, seed_grid,
                               annotation_sigma_px=1.0, *, triangles=None) -> ProjectionInverse: ...
def build_projection_triangles(seed_grid) -> ProjectionTriangles: ...

class SphereSurfaceContact:
    def __init__(self, surface, camera, seed_grid): ...
    def compute(self, center_px, depth_mm, radius_mm, side): ...
def intersect_sphere_rays(ray_origins, ray_directions, sphere): ...
def select_sphere_cap(candidate_points, reference_point, reference_normal,
                       side, position_tolerance_mm): ...
def sphere_normals(points, sphere): ...

def load_regions(path, image_root=None): ...
def save_regions(path, regions, *, image_root): ...
def region_boundary_points(region, count=128): ...
def region_masks(region, image_shape, *, outer_margin_px): ...
def edit_regions(regions, *, panel_callback, recompute_callback, save_callback,
                 editable_panel=0, update_geometry=None, indentation_range_mm=None): ...
```

| 对象/返回值 | 契约 |
|---|---|
| `CameraModel` | matrix、distortion、image_shape；相机坐标系 |
| `ProjectionInverse` | parameters、xyz、valid、sigma_mm、reprojection_error_px |
| seed_grid | parameters、xyz、uv、几何 mask；不要求原生参数行列均匀 |
| `Sphere` | center、radius_mm |
| `SphereContact` | sphere、center_parameters、center_frame `(A,e1,e2,n0)`、center_sigma_mm、boundary_parameters、boundary_xyz、boundary_px；side 为 ±1 |
| 射线求交返回值 | 按 minus/plus 顺序的 t、交点、valid；选冠返回固定 root_sign |
| `sphere_normals` | 球外法向，胶皮方向因子由标定适配层施加 |
| `RegionAnnotation` | image、image_size、status、center_px、shape、semi_axes_px、angle_degrees、contour_px、semantics、indentation_depth_mm |

通用区域模块不认识球心和相机，通用球模块不认识图片和标注文件，编辑器不导入标定 pipeline。接触适配器设置 `semantics=contact_boundary`，传入边距并组装回调。`ContactAnnotation` 可保留为通用区域类型的兼容别名。

### 8.5 标定编排、LUT 与输入适配接口

```python
def prepare_reference(config_path, settings): ...
def prepare_image(config_path, settings, reference, image_path): ...
def load_frame_calibration_data(path, settings): ...
def finalize_image(prepared, reference, annotation, settings): ...
def compute_frame_colors(final_frame, reference, settings): ...
def sphere_contact_for_annotation(frame, annotation, settings): ...
def make_sphere_labels(frame, annotation, sphere, settings): ...
def export_normal_samples(frame_records, settings, output_path): ...
def fit_and_validate_samples(sample_path, settings): ...
def update_material_sampling_grid(settings: CalibrationSettings) -> Path: ...
def generate_calibration_visualizations(calibration_file, sample_file, work_dir,
                                       output_dir=None, *, color_distance_threshold=0.01,
                                       normal_angle_threshold_degrees=20.0) -> Path: ...

def build_normal_lut(colors, slopes, *, size=64,
                     maximum_rms_angle_degrees=5.0) -> NormalCalibration: ...
def lookup_slopes_raw(color_residual, calibration, valid_mask=None): ...
def find_ambiguous_color_pairs(colors, slopes, *, color_distance_threshold=0.01,
                               normal_angle_threshold_degrees=20.0): ...
```

`FrameCalibrationData` 是编排层数据，包含图片身份、原图、区域 mask、跟踪观测、模板与参考 marker 尺寸、曲面系数和适配器、camera、chart、chart_file、三套坐标、geometry_stage、fit_info。底层函数只接收从中拆出的数组或通用对象。

加载缓存时用模板和系数恢复曲面适配器，不能用缓存 XYZ 重新拟合系数。工作流使用 `FrameCalibrationError(code,message)` 记录失败，底层数学函数不决定跳过图片或写报告。

`lookup_slopes_raw` 返回 slopes、variance、confidence。数据结构和原始三线性插值移入 normal_mapping；`recon.lookup_slopes` 保留原签名，包装零色差衰减与可信无接触策略。原模块重导出旧 API，不复制实现。

`NormalCalibration.metadata` 保存坐标图、基约定和验证状态，load/save 均保留。旧 v4/v5 数值格式可读，但缺少新声明不能自动用于新坐标路径。

`get_surface.save_material_coordinates` 增加关键字参数 `surface_coefficients=None`、`geometry_stage="final"`、`metadata=None`。元数据不得覆盖数组含义。`recon._load_reconstruction_input(path,expected_metadata=None)` 保持返回五元组：XYZ、曲面色差、几何有效域、光度有效域、可信无接触域。

## 9. 执行阶段与配置

以下阶段已由 `calibrate-norm` 提供，具体数值校验以当前实现为准。

| 阶段 | 输入 | 输出 | 界面 |
|---|---|---|---|
| prepare | 原图、底色、相机、模板 | reference、坐标图、prepared、manifest | 无 |
| annotate | prepared、球半径和球心侧 | 标注文件、最终几何预览 | 缺标注或 --edit 时 |
| samples | prepared、accepted 标注 | 最终整体曲面、颜色/标签、聚合样本、逐图报告 | 无 |
| fit | 聚合样本、同批 reference 几何/坐标图及 frames/overall 缓存 | 训练 LUT、验证报告、通过后的最终 LUT 和新采样网格 | 无 |
| all | 上述源输入 | 按顺序执行全部阶段 | 仅标注阶段 |

`prepare` 不要求球半径；`annotate/samples` 要求明确的几何输入。`samples/fit` 不初始化 GUI，fit 不读取原图。

沿用现有配置键，在 `normal_calibration` 下增加图片模式设置：

```yaml
normal_calibration:
  input_mode: images
  images: ["assets/cali_norm_pics/*.npz"]
  reference_image: assets/reference.npz
  sphere_radius_mm: null             # 必填：真实半径
  sphere_side: null                  # positive_normal / negative_normal
  boundary_model: zero_boundary_displacement
  annotations_file: assets/normal_calibration/manual_regions.yaml
  work_dir: assets/normal_calibration
  input_file: assets/normal_calibration/normal_samples.npz
  output: assets/normal_calibration/normal_lut.npz
  lut_size: 64
  maximum_rms_angle_degrees: 5.0
  checks:
    background_max_abs: 0.01
    background_min_fraction: 0.1
    validation_fraction: 0.2
    validation_rms_degrees: 5.0
    maximum_out_of_range_fraction: 0.05
```

旧样本模式默认值仍为 `input_mode: samples`。图片、底色路径、半径和球心侧必须来自本次数据，不从旧标注猜测。

CLI 保留 `--config / --input / --output`，新增 `--stage`、`--images`、`--reference-image`、`--sphere-radius-mm`、`--sphere-side`、`--annotations`、`--edit`。显式 `--input` 选择样本模式，与 `--images` 互斥；图片模式默认 all，样本模式默认 fit。

```bash
uv run calibrate-norm --config config.yaml --stage prepare
uv run calibrate-norm --config config.yaml --stage annotate
uv run calibrate-norm --config config.yaml --stage samples
uv run calibrate-norm --config config.yaml --stage fit
```

`--input` 继续覆盖现有 `input_file` 配置键。图片按解析后的路径去重排序，生成六位索引缓存名，manifest 固定保存 index、key、源路径和参考路径。标注变化后从 samples 重跑；坐标图、模板或相机变化后从 prepare 重跑。

## 10. 文件与数据约定

### 10.1 输出布局

原始拍照图像保存在 `assets/cali_norm_pics/`，默认生成产物集中在 `assets/normal_calibration/`。修改 `work_dir` 时，未显式指定的标注、样本和 LUT 路径跟随该目录；图片扫描会排除整个产物目录，递归 glob 不会重新读入缓存。

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
  verification/report.csv
  verification/validation.json
```

每图输出缓存和一行报告，不自动导出检查 PNG。标注窗口直接显示线性原图、轮廓/marker 屏蔽、曲面色差/sample mask、球面法向/拟合结果；反投影图仅用于显示。几何缓存 v3 保存 `observed_linear_rgb` 浮点整图；旧 v2 缓存在读取入口解码。

### 10.2 NPZ 字段

数值几何用 float64；保存的颜色、坡度和普通采样网格用 float32；mask 用 bool；标量字符串和路径用 Unicode。所有文件用 `allow_pickle=False` 读取，不保存 object 数组。

| 文件 | 版本与必需内容 |
|---|---|
| reference_coordinates | v1；chart_id、coordinate_kind、parameter_bounds、a_axis、b_axis、st_table、arc_lengths_mm、row_lengths_mm、column_lengths_mm；数值全部 float64 |
| reference_surface / prepared / overall | v4；保留现有三套 XYZ/UV/coordinates/mask；新增各套 st、surface_coefficients(16×3)、camera_matrix、dist_coeffs、material_bounds_mm、template_xy_mm、image_shape、source_image、geometry_stage、surface_mask、principal_curvatures_per_mm、curvature_valid_mask |
| prepared 附加 | image_key/index、marker_pixels/valid/confidence、reference_marker_sizes、marker_image_mask；足以重新拟合，不仅保存渲染结果 |
| overall 附加 | marker_fit_valid、最终系数与 fit_info；geometry_stage=final |
| frames | xyz/uv/st/material_coordinates、reference_color_grid、observed_color_grid、color_residual_grid、geometry_valid_mask、photometric_valid_mask、sample_valid_mask、slope_grid、normal_gt_grid、sphere_center_xyz、sphere_radius_mm、sphere_root_sign、sphere_side、boundary_model、indentation_depth_mm、contact_center_parameters、contact_boundary_px、principal_curvatures_per_mm、curvature_valid_mask |
| normal_samples | v1；color_residual_samples(N×3)、slope_samples(N×2)、image_index(N)、material_coordinates(N×2)、st_samples(N×2)、image_paths(M)、reference_image、sphere_radius_mm、sphere_side、boundary_model |
| LUT | 沿用 v5 signed 数值格式；metadata 增加参考背景、坐标与基约定、几何模型、validation_status 和验证分组/指标 |

所有几何、颜色和样本产物包含 `coordinate_kind=reference_arclength`、`coordinate_chart_id`、`coordinate_chart_file`、`frame_convention=st_grid_columns_rows`。颜色/图像样本另含 `residual_domain=material_grid`、`color_space=linear_rgb`、`rgb_transfer_function=bt709`；图像 LUT metadata 同样记录曲线。

`*_coordinates` 始终保存反解后的原生 a,b；`*_st` 保存规则弧长网格。保留 observation 的 `xyz/uv/valid_mask` 别名，新增 `coordinates/st` 别名；`valid_mask` 与新增 geometry_valid_mask 均表示几何域。fit_info、metadata 等字典在 NPZ 中存为 Unicode JSON，不使用 object 数组。

坐标图单独保存，不在每个文件重复积分表。chart_file 相对当前 NPZ 所在目录；聚合或搬到另一目录时重新计算相对路径。普通 get-surface 自动建图时保存为 `<output_stem>_coordinates.npz`。

本流程要求带接触面偏移声明的 v4 几何缓存；更旧的缓存必须重新 prepare。v4 中旧 `preview_reference` 可从现有观测升级为当前图初始拟合，只有收敛后的 `final` 可生成训练样本。

### 10.3 标注 YAML

```yaml
version: 2
image_root: ../cali_norm_pics
regions:
  - image: capture_0001.npz
    image_size: [640, 480]          # width,height
    status: accepted
    semantics: contact_boundary
    center_px: [320.0, 240.0]
    shape: sphere_contact
    indentation_depth_mm: 0.3
    contour_px: [[300.0, 240.0], [320.0, 260.0], [340.0, 240.0], [320.0, 220.0]] # 示例；实际由曲面求解
```

image_root 相对 YAML 所在目录解析。region 的 image_size 与实际图片不符时拒绝；旧格式只在几何语义明确时显式转换，不自动继承旧图片路径。

## 11. 最少校验与开发验收

运行校验合并为四类，不再增加独立人工检查步骤：

| 类别 | 通过条件 |
|---|---|
| 整体几何 | 弧长积分收敛、坐标图可逆与反解有效、跨文件坐标身份匹配；复用 provider 的 ID/单位和拟合门槛。当前为至少 32 个有效 marker、两方向覆盖 ≥75%、重投影 RMS ≤1.5 px、度量误差 ≤0.08，ambiguous 拒绝；实际阈值读取 provider 配置 |
| 固定背景 | 在两侧光度有效且接触外扩区之外统计；背景占两侧光度有效点比例 ≥0.1；`median(max(abs(delta),RGB))≤0.01`。全部在曲面域计算，无点不能通过 |
| 球面真值 | 交线闭合与投影有效、marker 排除集合收敛、唯一球冠分支、法向方向和有限坡度均通过，且最终有有效配对样本 |
| LUT 整图留出 | 训练/验证按图片隔离；每张验证图角 RMS ≤5°，范围外比例 ≤0.05；通过才发布最终 LUT |

表中的背景和验证阈值读取 checks，LUT 节点阈值沿用原算法。数组形状、有限值和文件字段是入口契约，不额外生成一套验收流程。

逐图报告至少保存 image_key、stage、status、reason、有效样本数及前三类指标。失败 reason 使用稳定代码：`region_not_found / tracking_failed / geometry_rejected / coordinate_chart_invalid / background_mismatch / annotation_missing / sphere_fit_failed / sphere_branch_ambiguous / no_valid_samples`。被跳过图片不混入训练；验证失败单独记在 validation.json。

开发验收只写四组有意义的场景，放在 `tests/test_norm_calibration_pipeline.py`：

1. **弧长与坐标闭环**：抛物线长度区别于线性参数，等速圆柱正确退化；二维耦合图正反解、保存加载、逆投影闭环；冻结参考 st 后当前伸缩不改变材料身份；拒绝不可逆图和多解。
2. **独立颜色采样与缺失区**：两张不同 UV 的解析梯度图在同一曲面颜色相同，差值为零；两侧 marker、饱和与邻域越界正确排除；marker 位移未知量不变成零边界。
3. **球面几何与符号**：平面接触半径解析解、倾斜/畸变、弯曲非圆轮廓、两侧及 d=R；截断/多交点与分支不定会失败；法向↔坡度往返一致。
4. **无界面全链路与复用**：至少三张合成分组数据完成 samples/fit；get-surface 与标定共用函数结果一致，非双三次曲面可接入；验证图不影响训练范围；验证失败保留旧 LUT；旧样本模式明确未验证。

采用标准库 unittest。测试覆盖上述行为，不逐行复刻实现。实际数据运行后查看交互标注和结构化报告，不依赖检查 PNG。

## 12. 数值细节与实施顺序

集中使用以下默认值，避免散落在 GUI、provider 和数学函数中：

| 环节 | 数值细节 |
|---|---|
| 弧长逆解 | Newton 最多 16 次、回溯 8 次、残差 ≤1e-8；只允许 ≤1e-12 的舍入越界回到边界 |
| UV 三角形初值 | 四边形沿左上—右下对角线拆分；重心权重容差 -1e-6，面积绝对值 <1e-8 px² 跳过 |
| 曲面逆投影 | least_squares 最多 50 次评估，ftol/xtol/gtol=1e-10；原生归一化参数按 1e-4 去重；同变量差分步长 1e-5，边界单边差分 |
| marker 光度屏蔽 | 搜索半径至少为参考 bbox 半对角线+2 px，并取不小于 ceil(0.35×中位 marker 间距)；中心匹配唯一组件，否则用该保守半径圆盘；结果外扩 1 px（3×3 核） |
| 球面判别式 | 小于 -1e-9 mm² 无效；仅将此容差内的小负值归零 |
| 接触区域 | 128 方向闭合交线；使用最终轮廓，不内缩；外扩 20 px 仅用于拟合及背景排除；排除集合最多迭代 6 次 |

按依赖顺序实施：

| 顺序 | 完成内容 | 可检查结果 |
|---|---|---|
| 1 | 黑框区域输出、通用曲面/采样/弧长图 | 实际轮廓 mask；st 与原生参数分开；同图跨分辨率复用 |
| 2 | provider 拆分跟踪和拟合，接通 get-surface | 独立照片固定 ID；预览与最终阶段可区分；无需独立分割后端 |
| 3 | 曲面颜色、marker 屏蔽和缓存 | 两侧独立采样；几何/光度 mask 分开；缓存可回放 |
| 4 | 通用区域编辑器和接触适配 | 可保存、续标、跳过、重算；accepted 标注可无界面复用 |
| 5 | 曲面逆投影、球心和法向标签 | 可复用几何模块；同 UV 的颜色与法向成对 |
| 6 | 样本、CLI、LUT 验证与发布 | 分阶段重跑；整图隔离；失败不覆盖最终 LUT |
| 7 | recon-local 适配、四组验收、文档同步 | 推理与训练定义一致；marker 孔洞可推断；更新 docs/structure.md |

实施前需要填入的外部信息只有：本次图片和底色路径、真实球半径、球心所在法向侧，以及零边界位移模型是否适用。本文不依赖旧数据目录是否仍存在。
