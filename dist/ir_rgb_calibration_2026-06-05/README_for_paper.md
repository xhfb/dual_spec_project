# 红外→可见光 空间配准（单应标定）数据包说明

本数据包用于支撑论文中「红外(IR)与可见光(RGB)双相机空间配准」一节的写作与数据归档。
内含完整的标定方法、原始对应点、采集图像、求解得到的单应矩阵与逐点误差。

**标定日期**：2026-06-05  
**工作距离 Z0**：1.5 m（元数据记录值，不参与 H 求解）  
**验收状态**：通过（`passed: true`）

---

## 1. 目标

求单应矩阵 H（3×3），把红外热成像像素坐标映射到可见光像素平面：

```
[x_rgb, y_rgb, 1]^T ∝ H · [x_ir, y_ir, 1]^T
```

用途：后续 Late Fusion 把在 IR 上检测到的目标框/点投影到 RGB 坐标系，做框级 IoU 关联（跨模态目标匹配）。

## 2. 硬件与坐标系

- **可见光相机**：LRCP 1080P USB 摄像头，标定取流分辨率 640×480 @ 60fps（MJPG）。
- **红外热像仪**：HIK 机芯，原生分辨率 256×192，经 libuvc 取流，输出 16-bit 温度矩阵 + YUYV 图像。
- **安装方式**：两相机上下并排刚性固定，红外在上、可见光在下，光轴朝向一致，**基线约 4cm**，未做任何出厂配准。
- **坐标系约定**：
  - IR：256×192，且已按固定 `roll` 偏移对齐（`roll_x = -164, roll_y = -2`）。H 的输入坐标即此「已对齐」空间。
  - RGB：640×480。H 的输出坐标即此空间。

## 3. 标定板

- 平面板上 9 个加热点，排成 3×3，点间距约 10~12cm。
- 每个加热点用直径约 1cm 的红色圆点覆盖：在红外中表现为热亮斑，在可见光中表现为白底红圆点。
- 板上的 ArUco 角标与色卡在红外中不可见，**未用于 IR↔RGB 对应**；对应点只取红点/热斑。

## 4. 采集与同步

- 双路独立采集线程各自抓取「最新帧」并打 `time.monotonic()` 时间戳。
- 取帧对时要求两路时间戳差 |Δt| ≤ 33ms，否则判为不同步、丢弃。
- 操作员把标定板摆到画面不同位置（左上/右上/左下/右下/中心等），每个位置抓取一对同步帧。
- 采集时移除画面中的人与其它热源，保证背景冷且均匀。
- **本次标定在工作距离约 1.5 m 下进行**，各组仅变化板在画面内的位置，尽量保持深度一致。

## 5. 对应点标注（人工）

- 网页端并排显示 RGB 与 IR；点「保存帧对」后画面定格，便于精确点选。
- 对每个位置，按**固定顺序（左→右、上→下）**依次点选：先点 IR 热斑、再点对应 RGB 红点。
- 点选在各自原始分辨率坐标系下进行（IR 256×192，RGB 640×480）。
- **本次部分组因遮挡/视野限制未标满 9 点**（见下表），但仍满足总点数 ≥30 的验收要求。

## 6. 单应求解与误差度量

- 汇总所有组的对应点，用 OpenCV `cv2.findHomography(pts_ir, pts_rgb, cv2.RANSAC, ransacReprojThreshold=3.0)` 求解 H。
- 误差度量：把全部 IR 点经 H 投影到 RGB，计算每点欧氏重投影误差（像素），并统计总体 RMSE。
- 离群判定：RANSAC 外点，或重投影误差 > 3px。

## 7. 本次标定结果

| 指标 | 数值 |
|------|------|
| 采集组数 | **6 组** |
| 总对应点数 | **50 点** |
| 总体重投影 RMSE | **2.48 px** |
| RANSAC 内点 RMSE | **1.73 px** |
| RANSAC 内点 / 离群 | 35 / 16 |
| 最大单点误差 | 5.59 px |
| 标定参考距离 Z0 | **1.5 m** |
| 验收（RMSE ≤ 5px 且 ≥4 组 且 ≥30 点） | **通过** |

逐组重投影误差（基于最终 H，见 `results/per_group_summary.csv`）：

| 组 | 帧对 | 点数 | RMSE(px) | 最大误差(px) | ≤3px 点数 |
|----|------|------|----------|--------------|-----------|
| 0 | pair_00 | 9 | 2.14 | 3.42 | 7 |
| 1 | pair_01 | 9 | 2.26 | 5.59 | 8 |
| 2 | pair_03 | 6 | 2.24 | 4.67 | 5 |
| 3 | pair_04 | 9 | 2.18 | 2.91 | 9 |
| 4 | pair_05 | 8 | 2.89 | 4.99 | 6 |
| 5 | pair_06 | 9 | 2.99 | 4.51 | 5 |
| **合计** | — | **50** | **2.48** | **5.59** | **40** |

> 注：组 2 关联帧对为 `pair_03`（`pair_02` 已跳过未纳入对应点）。组 2 仅标 6 对点（视野内可辨别的热斑/红点不足 9 个）。

完整逐点数据见 `results/reprojection_errors.csv`，单应矩阵数值见 `results/homography_matrix.txt` 与 `results/homography.npy`。

## 8. 与历史标定（Z0=0.6m）的对比

| 项目 | 2026-06-05 早期（0.6m） | **本次（1.5m）** |
|------|------------------------|------------------|
| 组数 / 点数 | 4 / 36 | 6 / 50 |
| 总体 RMSE | 4.39 px | **2.48 px** |
| 内点 RMSE | 1.62 px | **1.73 px** |
| 验收 | 通过 | **通过** |

在 1.5 m 工作距离下，4 cm 基线引入的视差更小，全局单应拟合更稳定；本次总体 RMSE 明显优于近距离标定。

## 9. 运行期使用（接口）

```python
from registration import load_homography, project_point, project_box
from types import DetectionBox

H = load_homography()                         # 加载 results/homography.npy
xy_rgb = project_point((x_ir, y_ir), H)       # IR 点 -> RGB 点
box_rgb = project_box(DetectionBox(x1,y1,x2,y2), H)
```

注意：单应把矩形映射成一般四边形，`project_box` 主返回 4 角变换后的**轴对齐外接框**，并附精确四边形角点。

## 10. 文件清单

```
README_for_paper.md          本说明（论文写作 / 归档用）
README_ARCHIVE.md            给下游 Agent 的快速索引

results/
  homography.npy             最终单应矩阵 (3x3 float64, IR->RGB)
  homography_matrix.txt      同上的可读文本
  homography_meta.json       元数据(日期/组数/点数/Z0/RMSE/内外点/分辨率等)
  reprojection_errors.csv    逐点重投影误差(含 ir/rgb/投影坐标)
  per_group_summary.csv      逐组误差汇总

data/
  correspondences.json       全部对应点原始记录(各组点对+帧对名+时间戳)
  overlay.png                误差可视化(RGB上: 绿=点选RGB, 红/橙=IR投影, 连线+误差)
  pair_0X_rgb.png            各组可见光原图(640x480)
  pair_0X_ir.png             各组红外可视化原图(256x192, 已roll对齐)
  pair_0X_ir_temp.npy        各组红外原始温度矩阵(256x192 uint16, 备查)

code/
  calibrate_homography.py    交互式标定工具(网页端)
  registration.py            运行期配准接口
  types.py                   DetectionBox 数据类型
  README_calibration.md      标定工具使用说明
```

## 11. 复现要点

- 标定取流：RGB 640×480@60(MJPG)，IR 经 uvc_demo 输出 temp+yuv、套用固定 roll 对齐。
- 关键参数：同步阈值 33ms；RANSAC 重投影阈值 3.0px；`--z0 1.5`。
- 验收阈值：总体 RMSE ≤ 5px、组数 ≥4、汇总点数 ≥30。
