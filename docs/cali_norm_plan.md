# 压球图片的色差—法向标定实施计划

本文规定待实施的流程、算法和接口，目标是从一批压球图片、固定底色图、已知球半径和接触边界标注，生成可供局部重建使用的色差→法向映射。文中“新增”“调整”均为待办，不代表代码已经实现。

## 核心功能与完整链路

标定需要完成三件事：重建胶皮整体曲面并建立公共曲面坐标；在公共坐标上取得色差和球面法向；将成对样本训练成 LUT，并用整张留出图片验证。

```text
固定底色图 → 黑框内部区域、marker → 参考整体曲面 Q_ref
                                         ↓
                                 参考弧长坐标图 F_ref
                                         ↓ 全批次共用
压球图片 → 黑框内部区域、marker 跟踪 → 整体曲面预览
                                         ↓
                                人工标注接触边界
                                         ↓ 排除接触区 marker
                                   最终整体曲面 Q_i
                                         ↓
                   同一规则 s,t 网格 → 原生参数 a,b → XYZ、UV
                              ↙                         ↘
          分别采样压球图和固定底色图              标注边界 + Q_i + 球半径
                     ↓                                  ↓
           曲面域线性 RGB 有符号差                 球心、球冠、法向
                              ↘                         ↙
                             色差—局部坡度成对样本
                                         ↓
                                LUT → 整图留出验证
```

**色差只在曲面域计算。** 当前图与底色图分别经各自的 UV 采样到同一个材料网格，再相减。黑色 marker 参与几何重建，但不提供颜色样本。

当前已有 marker 检测与跟踪、黑框内部区域检测、双三次整体曲面、线性 RGB 转换和 LUT 聚合。需要补全区域 mask 输出、弧长坐标图、接触排除后的重拟合、标注与球面标签、图片批处理，以及推理端的相同采样约定。

以下第 1—7 节说明功能，第 8 节确定代码结构，最后给出阶段、文件、校验和实施顺序。

## 1. 从图片重建整体曲面

### 1.1 复用 marker 检测得到胶皮区域

复用 `reconstruction/surface_reconstruction/markers.py:find_inner_roi` 的黑框检测和候选选择算法。当前只返回矩形 ROI；新增结果应同时提供选中的内部轮廓、填充轮廓得到的 `surface_mask` 和 marker 搜索 ROI。

实现要求：

- `surface_mask` 是黑框内部的完整胶皮区域，包含 marker 所在位置；不要用矩形填充代替实际轮廓。
- `DETECT_X_MIN / DETECT_X_MAX` 限制原图像素中的 marker 搜索横坐标范围（包含上下界），不应裁掉胶皮轮廓和 mask；检测坐标仍为原图坐标。
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

### 1.3 压球图：先跟踪和标注，再最终拟合

独立照片不按视频的连续帧处理。每张图都从同一参考图初始化：

1. 绑定参考弧长图，重置跟踪状态，处理参考图以恢复固定 ID。
2. 跟踪到压球图，只记录观测，不拟合包含接触 marker 的整体曲面。
3. 用参考曲面系数生成标注预览，标记为 `preview_reference`。
4. 标注接触边界；将其向外扩张得到接触排除区。
5. 从缓存观测中排除接触区 marker，以参考系数为先验重新拟合，得到 `Q_i`。
6. 用最终系数重新计算 XYZ、UV、颜色和球面标签。预览数据不能直接训练。

为此必须把 session 的“跟踪”和“拟合”拆开。排除接触点只修改 `marker_fit_valid`，不删除它们的跟踪位置，因为颜色屏蔽仍需要这些位置。真正的跟踪失效与拟合异常继续沿用现有处理。

重复调整标注时，重用同一份跟踪缓存，不重复执行光流更新。`fit=False` 也不应增加拟合失败计数。

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
底色 BGR8 → 线性 RGB → 按 UV_ref 采样 → C_ref(s,t)
压球 BGR8 → 线性 RGB → 按 UV_i   采样 → C_i(s,t)
delta(s,t) = C_i(s,t) - C_ref(s,t)
```

保留有符号的三个通道，不取绝对值、不转灰度、不做显示归一化。固定底色假设由最终曲面上的非接触区色差验证，不因“形变近似为零”省掉曲面坐标转换。

参考、当前网格的 chart_id、st、原生 parameters 必须一致；各自使用自己的 XYZ、UV 和有效域。

### 3.2 黑色 marker 只造成光度缺失

每张原图先建立：

```text
source_valid = surface_mask & ~marker_image_mask & unsaturated
```

`unsaturated` 要求原始 BGR8 三个通道均小于 255。marker mask 由固定 ID 的位置和局部黑色连通域取得，包含组件形状及边缘保护带；参考图和当前图分别建立。

复用现有组件检测，但保持 `detect_cross_centers` 的三个返回值兼容。提取公共的组件分析辅助函数，避免再写一份阈值和连通域算法。组件对应不唯一、暂时失检时，按参考尺寸和已跟踪位置保守屏蔽。搜索半径与边缘扩张值见第 12 节。

颜色采样采用严格四邻域双线性插值：四个源像素都在图像内、source_valid 且有限才有效。图像最后一行/列不具备完整邻域，不能复制边界凑齐。两张图分别采样后，将两侧有效性与几何有效域相交，得到 `photometric_valid_mask`。

不对 marker 黑色区域插值补色后训练；也不把它当作零色差、零法向变化或零位移。几何 mask 始终保留这些位置。

## 4. 可视化标注接触区域

### 4.1 标注内容与界面

每张图显示三个同步面板：

| 面板 | 内容 |
|---|---|
| 原图 | 黑框、marker 屏蔽区、同步显示的接触轮廓与中心 |
| 曲面色差 | 映射回原图像素位置的曲面域 RGB 色差、有效域、最终 sample mask；直接编辑接触轮廓和中心 |
| 几何预览 | 整体曲面、球面法向/坡度、拟合误差或失败原因 |

在中间色差图上按圆形接触边界标注，只编辑圆心和半径；拖动圆周及滚轮调整半径始终保持圆形，滚轮不缩放画面，不提供椭圆半轴调整、旋转或多边形转换。左侧原图同步显示轮廓供对照。三栏宽度为 640、960、280 像素，图像区高 720 像素，各图等比例显示；说明文字采用 22 像素字号和 30 像素行距，独立于源图绘制并按实际字宽换行，底部预留四行状态区。标注窗口与保存预览使用相同布局，避免文字随源图反复缩放而模糊。色差图必须映射回原图像素位置，鼠标坐标需扣除面板偏移并除以显示缩放；保存原图像素坐标。旧椭圆与多边形标注保留读取兼容，重新打开编辑时以相同圆心和等面积圆初始化，重新确认后保存。

标注语义固定为 `contact_boundary`：胶皮与球接触区域的边界。中心是接触区内部的代表点，不是球心投影。普通 ROI、可见球轮廓不能自动充当接触边界。

已实现标注圆物理尺寸上限：取有效曲面的最大正相机 `Z_max`（mm），将配置中的球半径 `R`（mm）换算成原图像素半径上限。无畸变时为 `r_max = min(fx, fy) * R / Z_max`；实际实现用含畸变的相机模型，在当前圆心对应的 `Z_max` 平面投影半径 `R` 的相机平行圆盘，128 个均匀边界方向到圆心的最小像素距离即为上限。初始使用 prepared 曲面，重算后更新为最近得到的最终曲面。初始圆、拖动、滚轮和撤回均裁到该上限，面板显示当前及最大直径；重算引起收缩时清除过期结果，重新计算通过才可接受。没有有效正深度时报错。共用投影计算放在 `utils/surface_projection.py:projected_circle_radius_px`，通用编辑器只接收半径上限回调，不依赖标定状态。

按键固定：`g` 重算当前标注的最终几何与标签；`Enter` 重算通过后接受并下一张；`x` 标记跳过；`Esc` 保存草稿并退出。失败显示原因，保留草稿。已有 accepted 标注默认复用，只有 `--edit` 才重新编辑。

### 4.2 标注转换为边界和 mask

- 圆形边界按均匀角度采样得到 128 点；兼容读取的旧椭圆按均匀角度、多边形按边长累计弧长采样。
- 原轮廓填充得到接触区；向内收缩 3 px 得到标签采样区。
- 原轮廓向外扩张 20 px 得到整体拟合与背景统计的排除区。
- 图像 mask 经各自 UV 转到曲面网格，使用统一的保守邻域有效性规则。

标注保存为 YAML，逐图状态为 `draft / accepted / skip`；图片 key 使用相对统一 image_root 的完整路径，不用 stem。只修改标注时可从 prepared 缓存重算，无需重新跟踪。文件字段见第 10 节。

## 5. 从标注与整体曲面计算球面法向

### 5.1 明确几何假设

输入必须给出真实正半径 R、球心相对参考法向的一侧，以及 `boundary_model=zero_boundary_displacement`。

该模型认为接触边界处的局部位移近似为零，所以标注边界的三维点同时位于整体曲面和已知半径球面上。若实际数据不满足这一模型，仅靠轮廓和整体曲面不能可靠确定球心，应补充球心位姿或深度约束后另行扩展；程序不能猜测。

本节的“参考法向”指当前整体曲面 Q_i 的局部基准法向，不是球面法向。所有 XYZ、球心和射线均在相机坐标系，长度统一为 mm。

### 5.2 将边界像素反解到整体曲面

对最终网格的有效四边形各建两个 UV 三角形，用重心坐标给出边界像素的原生参数初值；随后最小化：

```text
project(Q_i(a,b)) - boundary_pixel
```

优化内部使用归一化原生 alpha、beta，评估曲面时转换回 a、b。不能直接把 st 输入曲面模型。

接受重投影误差 ≤1 px 的解；多个三角形得到的相同解聚类去重，无解或多个不同解均无效，不外推。128 个边界点至少 96 个有效，循环轮廓连续缺失不超过 32 点。

同时返回三维标注误差尺度：

```text
sigma_mm = sigma_px * ||J_Q @ inverse(J_projection)||_2
```

两个 Jacobian 对同一内部参数求导，`sigma_px=1`，sigma_mm 下限 `1e-6`。不可逆时该点无效。接触中心也按相同方法反解，得到 A 与误差尺度；从最终 st 网格插值得到局部正交基 `e1,e2,n0`，重新正交归一。

### 5.3 固定半径拟合球心

将有效边界点 X 投到以 A 为原点的切平面，线性拟合圆：

```text
x = (X-A)·e1
y = (X-A)·e2
[x,y,1] [D,E,F]^T = -(x²+y²)
cx = -D/2
cy = -E/2
r_contact² = cx² + cy² - F
```

要求 `0<r_contact<R`。球心侧 `positive_normal / negative_normal` 对应 sigma=+1/-1：

```text
C(cx,cy,z) = A + cx*e1 + cy*e2 + sigma*z*n0
z_initial = sqrt(R²-r_contact²)
residual_j = (||X_j-C||-R) / sigma_mm_j
```

固定 R，使用带边界的 `least_squares`：cx、cy 在 ±2R，z 在 `[1e-6,2R]`；`loss="soft_l1"`、`f_scale=1`、最多 200 次评估。

要求优化成功、结果有限、归一化边界 RMS ≤2，且优化 Jacobian 最小/最大奇异值比 ≥`1e-6`。不通过时拒绝该图，不裁剪半径或强制接受球心。

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

将选中的 minus/plus 分支锁定到整张图，不逐点切根或翻转法向。每点还须位于原始标注圆盘内，不缩小半径或内缩边界，并满足 `n_gt·n0_center>0.1`、`n_gt·n0(point)>0.1`。

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

已实现标定后可视化：发布成功后，`calibrate/visualize_norm.py` 使用本批 LUT、聚合样本及 `frames/overall` 缓存，在 `work_dir/visualization/` 生成逐图色差与 p/q 对照、RGB 色差到 p/q 的三维映射、三个零通道 LUT 切片和整批汇总。逐图预测调用最终 LUT 的原始插值；独立留出统计调用同批 `_train.npz`，复用已有局部坡度转法向函数计算角误差。显示回投复用 `rasterize_surface_grid`，统计始终使用原材料网格样本。所有输入校验坐标身份；空间样本还检查与训练样本逐点一致，不混入遗留缓存。可视化失败不改写已发布 LUT 的验证状态；完整报告生成后才替换旧报告。没有图片分组的旧两数组模式不生成该报告。

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
| `utils/sphere_geometry.py` | 固定半径球心拟合、射线双根、球冠选择、球外法向 |
| `utils/region_annotations.py` | 区域数据、读写、边界采样、mask 与边距 |
| `utils/annotation_editor.py` | 通用面板和形状编辑；通过回调重算、保存 |
| `utils/annotation_window.py` | 固定图像范围的 Matplotlib/PyQt5 窗口，滚轮只转交给形状编辑 |
| `utils/color_difference.py` | BGR8/线性 RGB 转换、曲面数组有符号差、显示转换 |
| `utils/normal_mapping.py` | 提取 LUT 数据、构建与原始查询；保留兼容导出 |
| `calibrate/manual_norm_regions.py` | 为通用编辑器配置接触语义、三面板与重算回调 |
| `calibrate/norm_pipeline.py` | 图片准备、最终重拟合、颜色/标签组合、样本导出、验证 |
| `calibrate/calibrate_norm.py` | 保留现有 CLI 入口，解析参数并选择阶段 |
| `reconstruction/get_surface.py / reconstruction/local_recon.py` | 薄编排、文件适配和已有推理策略，不复制底层算法 |

依赖单向：底层数组与几何模块 → provider/标定适配 → CLI。禁止 `utils` 反向导入 `calibrate`，也不让数学函数读取配置、GUI 或 `provider.last_frame_result`。

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
def build_marker_image_mask(frame_bgr, marker_pixels, marker_valid,
                            reference_marker_sizes, *, dilation_px=1): ...

# Session
def process(self, frame, *, fit=True) -> dict: ...
def fit_observations(self, tracking, *, exclusion_mask=None,
                     prior_coefficients=None) -> dict: ...

# Provider
def bind_coordinate_chart(self, chart): ...
def reset(self, *, clear_coordinate_chart=False): ...
def track_frame(self, frame_bgr) -> dict: ...
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
                               annotation_sigma_px=1.0) -> ProjectionInverse: ...

def fit_sphere_center(boundary_xyz, sigma_mm, radius_mm, center_frame, side): ...
def intersect_sphere_rays(ray_origins, ray_directions, sphere): ...
def select_sphere_cap(candidate_points, reference_point, reference_normal,
                       side, position_tolerance_mm): ...
def sphere_normals(points, sphere): ...

def load_regions(path, image_root=None): ...
def save_regions(path, regions, *, image_root): ...
def region_boundary_points(region, count=128): ...
def region_masks(region, image_shape, *, outer_margin_px): ...
def edit_regions(regions, *, panel_callback, recompute_callback, save_callback,
                 editable_panel=0): ...
```

| 对象/返回值 | 契约 |
|---|---|
| `CameraModel` | matrix、distortion、image_shape；相机坐标系 |
| `ProjectionInverse` | parameters、xyz、valid、sigma_mm、reprojection_error_px |
| seed_grid | parameters、xyz、uv、几何 mask；不要求原生参数行列均匀 |
| `Sphere` | center、radius_mm |
| 球心拟合返回值 | Sphere、归一化 RMS；center_frame 为 `(A,e1,e2,n0)`，side 为 ±1 |
| 射线求交返回值 | 按 minus/plus 顺序的 t、交点、valid；选冠返回固定 root_sign |
| `sphere_normals` | 球外法向，胶皮方向因子由标定适配层施加 |
| `RegionAnnotation` | image、image_size、status、center_px、shape、semi_axes_px、angle_degrees、contour_px、semantics |

通用区域模块不认识球心和相机，通用球模块不认识图片和标注文件，编辑器不导入标定 pipeline。接触适配器设置 `semantics=contact_boundary`，传入边距并组装回调。`ContactAnnotation` 可保留为通用区域类型的兼容别名。

### 8.5 标定编排、LUT 与输入适配接口

```python
def prepare_reference(config_path, settings): ...
def prepare_image(config_path, settings, reference, image_path): ...
def load_frame_calibration_data(path): ...
def finalize_image(prepared, reference, annotation, settings): ...
def compute_frame_colors(final_frame, reference, settings): ...
def fit_ball_center(frame, annotation, settings): ...
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

以下是实施后应提供的行为，当前新增阶段尚不能执行。保留已有 `calibrate-norm` 入口。

| 阶段 | 输入 | 输出 | 界面 |
|---|---|---|---|
| prepare | 原图、底色、相机、模板 | reference、坐标图、prepared、manifest | 无 |
| annotate | prepared、球半径和球心侧 | 标注文件、最终几何预览 | 缺标注或 --edit 时 |
| samples | prepared、accepted 标注 | 最终整体曲面、颜色/标签、聚合样本、逐图报告 | 无 |
| fit | 聚合样本、同批 reference 几何/坐标图及 frames/overall 缓存 | 训练 LUT、验证报告、通过后的最终 LUT、新采样网格及可视化 | 无 |
| all | 上述源输入 | 按顺序执行全部阶段 | 仅标注阶段 |

`prepare` 不要求球半径；`annotate/samples` 要求明确的几何输入。`samples/fit` 不初始化 GUI，fit 不读取原图。

沿用现有配置键，在 `normal_calibration` 下增加图片模式设置：

```yaml
normal_calibration:
  input_mode: images
  images: ["data/cali_norm_pics/*.png"]
  reference_image: data/reference.png
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
  verification/000000.png
  verification/report.csv
  verification/validation.json
```

每图只输出一张合并预览和一行报告。预览包括轮廓/marker 屏蔽、曲面色差/sample mask、球面法向/拟合结果；反投影图仅用于显示。

### 10.2 NPZ 字段

数值几何用 float64；保存的颜色、坡度和普通采样网格用 float32；mask 用 bool；标量字符串和路径用 Unicode。所有文件用 `allow_pickle=False` 读取，不保存 object 数组。

| 文件 | 版本与必需内容 |
|---|---|
| reference_coordinates | v1；chart_id、coordinate_kind、parameter_bounds、a_axis、b_axis、st_table、arc_lengths_mm、row_lengths_mm、column_lengths_mm；数值全部 float64 |
| reference_surface / prepared / overall | v2；保留现有三套 XYZ/UV/coordinates/mask；新增各套 st、surface_coefficients(16×3)、camera_matrix、dist_coeffs、material_bounds_mm、template_xy_mm、image_shape、source_image、geometry_stage、surface_mask |
| prepared 附加 | image_key/index、marker_pixels/valid/confidence、reference_marker_sizes、marker_image_mask；足以重新拟合，不仅保存渲染结果 |
| overall 附加 | marker_fit_valid、最终系数与 fit_info；geometry_stage=final |
| frames | xyz/uv/st/material_coordinates、reference_color_grid、observed_color_grid、color_residual_grid、geometry_valid_mask、photometric_valid_mask、sample_valid_mask、slope_grid、normal_gt_grid、sphere_center_xyz、sphere_radius_mm、sphere_root_sign、sphere_side、boundary_model |
| normal_samples | v1；color_residual_samples(N×3)、slope_samples(N×2)、image_index(N)、material_coordinates(N×2)、st_samples(N×2)、image_paths(M)、reference_image、sphere_radius_mm、sphere_side、boundary_model |
| LUT | 沿用 v5 signed 数值格式；metadata 增加参考背景、坐标与基约定、几何模型、validation_status 和验证分组/指标 |

所有几何、颜色和样本产物包含 `coordinate_kind=reference_arclength`、`coordinate_chart_id`、`coordinate_chart_file`、`frame_convention=st_grid_columns_rows`。颜色/样本另含 `residual_domain=material_grid`、`color_space=linear_rgb`。

`*_coordinates` 始终保存反解后的原生 a,b；`*_st` 保存规则弧长网格。保留 observation 的 `xyz/uv/valid_mask` 别名，新增 `coordinates/st` 别名；`valid_mask` 与新增 geometry_valid_mask 均表示几何域。fit_info、metadata 等字典在 NPZ 中存为 Unicode JSON，不使用 object 数组。

坐标图单独保存，不在每个文件重复积分表。chart_file 相对当前 NPZ 所在目录；聚合或搬到另一目录时重新计算相对路径。普通 get-surface 自动建图时保存为 `<output_stem>_coordinates.npz`。

旧 v1 几何仍可供旧调用读取，但本流程缺系数/相机时必须重新 prepare。`preview_reference` 只能用于标注；samples 必须检查 final。

### 10.3 标注 YAML

```yaml
version: 2
image_root: ../../data
regions:
  - image: cali_norm_pics/0001.png
    image_size: [640, 480]          # width,height
    status: accepted
    semantics: contact_boundary
    center_px: [320.0, 240.0]
    shape: circle
    semi_axes_px: [40.0, 40.0]      # 两个值均为圆半径，单位 pixel
    angle_degrees: 0.0
    contour_px: null               # circle 不使用多边形顶点
```

image_root 相对 YAML 所在目录解析。region 的 image_size 与实际图片不符时拒绝；旧格式只在几何语义明确时显式转换，不自动继承旧图片路径。

## 11. 最少校验与开发验收

运行校验合并为四类，不再增加独立人工检查步骤：

| 类别 | 通过条件 |
|---|---|
| 整体几何 | 弧长积分收敛、坐标图可逆与反解有效、跨文件坐标身份匹配；复用 provider 的 ID/单位和拟合门槛。当前为至少 32 个有效 marker、两方向覆盖 ≥75%、重投影 RMS ≤1.5 px、度量误差 ≤0.08，ambiguous 拒绝；实际阈值读取 provider 配置 |
| 固定背景 | 在两侧光度有效且接触外扩区之外统计；背景占两侧光度有效点比例 ≥0.1；`median(max(abs(delta),RGB))≤0.01`。全部在曲面域计算，无点不能通过 |
| 球面真值 | 边界反解数量/连续性、球心精度与可确定性、唯一球冠分支、法向方向和有限坡度均通过，且最终有有效配对样本 |
| LUT 整图留出 | 训练/验证按图片隔离；每张验证图角 RMS ≤5°，范围外比例 ≤0.05；通过才发布最终 LUT |

表中的背景和验证阈值读取 checks，LUT 节点阈值沿用原算法。数组形状、有限值和文件字段是入口契约，不额外生成一套验收流程。

逐图报告至少保存 image_key、stage、status、reason、有效样本数及前三类指标。失败 reason 使用稳定代码：`region_not_found / tracking_failed / geometry_rejected / coordinate_chart_invalid / background_mismatch / annotation_missing / sphere_fit_failed / sphere_branch_ambiguous / no_valid_samples`。被跳过图片不混入训练；验证失败单独记在 validation.json。

开发验收只写四组有意义的场景，放在 `tests/test_norm_calibration_pipeline.py`：

1. **弧长与坐标闭环**：抛物线长度区别于线性参数，等速圆柱正确退化；二维耦合图正反解、保存加载、逆投影闭环；冻结参考 st 后当前伸缩不改变材料身份；拒绝不可逆图和多解。
2. **独立颜色采样与缺失区**：两张不同 UV 的解析梯度图在同一曲面颜色相同，差值为零；两侧 marker、饱和与邻域越界正确排除；marker 位移未知量不变成零边界。
3. **球面几何与符号**：已知球心、半径和接触圆，恢复两侧球冠及法向；错误边界与分支不定会失败；法向↔坡度往返一致。
4. **无界面全链路与复用**：至少三张合成分组数据完成 samples/fit；get-surface 与标定共用函数结果一致，非双三次曲面可接入；验证图不影响训练范围；验证失败保留旧 LUT；旧样本模式明确未验证。

采用标准库 unittest。测试覆盖上述行为，不逐行复刻实现。实际数据运行后仅查看每图合并预览与四类报告。

## 12. 数值细节与实施顺序

集中使用以下默认值，避免散落在 GUI、provider 和数学函数中：

| 环节 | 数值细节 |
|---|---|
| 弧长逆解 | Newton 最多 16 次、回溯 8 次、残差 ≤1e-8；只允许 ≤1e-12 的舍入越界回到边界 |
| UV 三角形初值 | 四边形沿左上—右下对角线拆分；重心权重容差 -1e-6，面积绝对值 <1e-8 px² 跳过 |
| 曲面逆投影 | least_squares 最多 50 次评估，ftol/xtol/gtol=1e-10；原生归一化参数按 1e-4 去重；同变量差分步长 1e-5，边界单边差分 |
| marker 光度屏蔽 | 搜索半径至少为参考 bbox 半对角线+2 px，并取不小于 ceil(0.35×中位 marker 间距)；中心匹配唯一组件，否则用该保守半径圆盘；结果外扩 1 px（3×3 核） |
| 球面判别式 | 小于 -1e-9 mm² 无效；仅将此容差内的小负值归零 |
| 接触区域 | 边界 128 点、至少 96 点有效、连续缺失 ≤32；使用原始标注圆盘，不内缩；外扩 20 px 仅用于拟合及背景排除 |

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
