# IR → RGB 空间配准标定（网页端）

把红外热成像（IR, 256×192）像素坐标，通过单应矩阵 `H` 映射到 RGB（640×480）像素平面，
供后续 Late Fusion 把 IR 检测框投影到 RGB 坐标系做框级 IoU 关联。

## 交付物

- `scripts/calibrate_homography.py` — 网页端交互标定（采集 + 手动点选 + 解 H + 误差报告）。
- `src/fusion/registration.py` — 运行期配准：`load_homography()` / `project_point()` / `project_box()`。
- `src/detection/types.py` — `DetectionBox`（最小占位，供 Late Fusion 复用）。
- 产物：`config/homography.npy`（3×3，IR→RGB）+ `config/homography_meta.json`。
- 采集中间件：`calib_data/`（帧对 png、IR 温度 npy、`correspondences.json`、`overlay.png`）。

## 环境

- 设备：`/dev/video_rgb`（LRCP 1080P，取 MJPG 640×480@60）、`/dev/video_ir`（HIK 热成像，经 `uvc_ubuntu/uvc_demo web` 取 temp+yuv）。
- 必须用**系统 python3**（已含带 GStreamer 的 cv2 4.10、fastapi、uvicorn、numpy）。
- IR 取流需 root：脚本内部用 `sudo -n ./uvc_demo web`，已确认本机 sudo 免密可用。
- IR 画面对齐沿用 `uvc_ubuntu/thermal_cam_debug.json` 的 `roll_x/roll_y`（与 `web_temp_viewer.py` 一致）。
- 不要同时运行 `web_temp_viewer.py`，否则会抢占 IR 设备。

## 运行

```bash
cd /home/sunrise/dual_spec_project
# 若当前用户对 IR 设备无权限，可整体用 sudo 跑（保证 uvc_demo 可启动）
python3 scripts/calibrate_homography.py --port 8001 --z0 0.6
```

浏览器打开 `http://<板子IP>:8001/`。

常用参数：`--sync-ms 33`、`--ransac-thresh 3`、`--z0 <板到相机平均距离米>`、
`--rgb-dev /dev/video_rgb`、`--ir-cmd "./uvc_demo web"`、`--calib-dir`、`--out-dir`、`--points 9`。

## 操作流程

1. 采集前：移除画面里的人和热源，背景尽量冷且均匀；标定板（3×3 加热点、红圆点覆盖）正对相机。
2. 把标定板摆到左上 / 右上 / 左下 / 右下 / 中心等位置（建议 ≥4~5 组，可换不同深度）。
3. 每个位置：
   - 等顶部「同步」徽标变绿（`|Δt| ≤ 33ms`），点【保存帧对】。
   - 按**固定顺序左→右、上→下**依次点 9 个点：先点 **IR 热斑**，再点对应 **RGB 红点**，凑成一对后点【确认/下一点】。
   - 点错可【撤销】；该位置不要可【跳过本组】。
   - 点满 9 点后点【完成本组】（写入 `calib_data/correspondences.json`）。
4. 采足后点【求解并保存】：RANSAC 解 H，页面回显 RMSE / 内点 / 离群，并展示 `overlay.png`。

## 验收

- 重投影 RMSE ≤ 5px；组数 ≥4~5、汇总点数 ≥30；
- `config/homography.npy` + `config/homography_meta.json` 落盘；
- `meta.passed` 为 true 表示三项硬指标均满足。

## 运行期使用（Late Fusion）

```python
from src.fusion.registration import load_homography, project_point, project_box
from src.detection.types import DetectionBox

H = load_homography()                       # 读取 config/homography.npy
rgb_xy = project_point((ir_x, ir_y), H)     # IR 点 -> RGB 点
rgb_box = project_box(DetectionBox(x1, y1, x2, y2, score, label), H)  # IR 框 -> RGB 轴对齐框
# rgb_box.quad 为 4 个变换后角点（精确四边形），rgb_box 本身为外接框，便于做 IoU
```

> 注意：`DetectionBox` 的 IR 坐标必须与标定时一致 —— 即 256×192 且已按 `thermal_cam_debug.json`
> 的 roll 对齐。后续 IR 检测器应输出同一坐标空间的框。
