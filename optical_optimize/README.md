# 胶皮光场匹配

从仓库根目录运行。光场对比和优化默认使用 `assets/reference_flat.npz`，相机参数和灯珠布局读自 `config.yaml`。实拍用相机 2、640×480 YUYV、手动曝光 20、白平衡 6500 K、背光补偿 0，灯带接 `/dev/ttyACM0`。

## 实拍结果

当前胶皮投影约 376×148 px，平面参考约 378×152 px。48 个标记的配准 RMS 为 0.70 px。以切换目标前的灯光为起点，通过 ACM0 逐灯调整 26 颗灯珠，再重放预设并独立采图。

| 对照图 | 优化前光场 RMSE | 重放后光场 RMSE | 降幅 |
| --- | ---: | ---: | ---: |
| `assets/reference_flat.npz` | 0.05029 | 0.04701 | 6.5% |

重放后 RGB 的全区均值偏差均小于 0.005。剩余差异主要是上半部红光偏弱、下半部偏强。本次验证覆盖当前构型下的无接触胶皮。[实拍对比图](runs/flat_final/comparison.png) · [数值报告](runs/flat_final/report.json)

## 采集、比较和调光

在胶皮无接触、位置固定时采集当前画面。这里的 `before` 是新建的采集目录。

```bash
uv run python -m optical_optimize capture --output optical_optimize/runs/before
uv run python -m optical_optimize compare optical_optimize/runs/before/current.npz \
  --output optical_optimize/runs/before_comparison
```

重新匹配时，程序先拍下当前状态，再从配置里的默认配色开始逐灯测量。可用 `--start-preset` 从已有预设开始，或用 `--reference` 指定另一张历史参考。优化结束后，胶皮保持实拍评分最低的灯光。

```bash
uv run python -m optical_optimize optimize --port /dev/ttyACM0
```

重新上电后可直接应用本次实拍确认的预设。

```bash
uv run python -m optical_optimize apply optical_optimize/reference_flat_preset.json --port /dev/ttyACM0
```

`apply` 校验灯珠布局，并将预设中的 RGB PWM 字节原样发送。当前布局为上、右、下、左各 9、4、9、4 颗，开头 3 颗保持熄灭。

从本次预设继续细调时，使用实拍采用的 PWM 上限和探测步长。

```bash
uv run python -m optical_optimize optimize --start-preset optical_optimize/reference_flat_preset.json \
  --max-pwm 140 --probe-step 24
```

上一轮以 `assets/reference.npz` 为目标的设置保存在 `reference_preset.json`。

## 比较和优化方法

1. 用胶皮的 48 个十字标记把相机画面变换到参考图坐标。比较区域取两图黑框内部的交集，避开黑框边缘和十字。报告记录对齐误差与区域覆盖率。
2. 在有效区按 12 行 × 28 列统计格内均值，以线性 RGB 的格均值 RMSE 为调光目标，同时记录逐像素误差、分区有符号偏差和接近饱和的比例。
3. 逐次改变一颗灯珠的一个 PWM 通道，实拍估计光场响应。每组测量前后用同一灯光补拍，补偿采集过程中的慢漂移。
4. 用带上下界的正则化最小二乘求 PWM 更新。每个候选都用相机实拍，保留误差下降的设置，并重新发送和采集最终设置。最终报告记录优化前后胶皮十字的位移。

`runs/` 保存本机记录。优化输出含 `before.npz`、`start.npz`、`after.npz` 及对应 PNG，另存 `comparison.png`、`report.json`、逐通道响应 `response.npz` 和可重放的 `preset.json`。NPZ 使用仓库统一的线性 RGB 格式，差值图以中灰表示零偏差。离线比较多张图时沿用第一张观测图求得的变换，适用于同一拍摄位置和胶皮构型。
