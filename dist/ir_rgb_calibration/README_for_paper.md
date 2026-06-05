# 红外→可见光 空间配准（单应标定）数据包说明

本数据包用于支撑论文中"红外(IR)与可见光(RGB)双相机空间配准"一节的写作。
内含完整的标定方法、原始对应点、采集图像、求解得到的单应矩阵与逐点误差。
以下为可直接引用的方法学描述、参数与结果。

---

## 1. 目标

求单应矩阵 H（3×3），把红外热成像像素坐标映射到可见光像素平面：

  [x_rgb, y_rgb, 1]^T ∝ H · [x_ir, y_ir, 1]^T

用途：后续 Late Fusion 把在 IR 上检测到的目标框/点投影到 RGB 坐标系，
做框级 IoU 关联（跨模态目标匹配）。

## 2. 硬件与坐标系

- 可见光相机：LRCP 1080P USB 摄像头，标定取流分辨率 640×480 @ 60fps（MJPG）。
- 红外热像仪：HIK 机芯，原生分辨率 256×192，经 libuvc 取流，输出 16-bit 温度矩阵 + YUYV 图像。
- 安装方式：两相机上下并排刚性固定，红外在上、可见光在下，光轴朝向一致，**基线约 4cm**，未做任何出厂配准。
- 坐标系约定：
  - IR：256×192，且已按固定 `roll` 偏移对齐（`roll_x = -164, roll_y = -2`，用于消除该机芯 YUYV 与温度矩阵的行列拼接错位）。H 的输入坐标即此"已对齐"空间。
  - RGB：640×480。H 的输出坐标即此空间。

## 3. 标定板

- 平面板上 9 个加热点，排成 3×3，点间距约 10~12cm。
- 每个加热点用直径约 1cm 的红色圆点覆盖：在红外中表现为热亮斑，在可见光中表现为白底红圆点。
- 板上的 ArUco 角标与色卡在红外中不可见，**未用于 IR↔RGB 对应**；对应点只取这 9 个红点/热斑。

## 4. 采集与同步

- 双路独立采集线程各自抓取"最新帧"并打 `time.monotonic()` 时间戳。
- 取帧对时要求两路时间戳差 |Δt| ≤ 33ms（约一帧），否则判为不同步、丢弃，不参与标定。
- 操作员把标定板摆到画面不同位置（左上/右上/左下/右下/中心等），每个位置抓取一对同步帧。
- 采集时移除画面中的人与其它热源，保证背景冷且均匀，避免红外干扰斑点。

## 5. 对应点标注（人工，主路径）

- 网页端并排显示 RGB 与 IR 实时画面；点"保存帧对"后画面定格，便于精确点选。
- 对每个位置，按**固定顺序（左→右、上→下）**依次点选 9 个点：先点 IR 热斑、再点对应 RGB 红点，构成一对 (ir_xy, rgb_xy)。
- 支持确认/撤销/跳过/完成本组，保证 IR 与 RGB 的点严格一一对应、同序。
- 点选在各自原始分辨率坐标系下进行（IR 256×192，RGB 640×480），无缩放误差引入。

## 6. 单应求解与误差度量

- 汇总所有组的对应点，用 OpenCV `cv2.findHomography(pts_ir, pts_rgb, cv2.RANSAC, ransacReprojThreshold=3.0)` 求解 H。
- 误差度量：把全部 IR 点经 H 投影到 RGB，计算每点欧氏重投影误差（像素），并统计总体 RMSE。
- 离群判定：RANSAC 外点，或重投影误差 > 3px。

## 7. 本次标定结果

- 采集：**4 组、共 36 个对应点**（每组 9 点）。
- 总体重投影 **RMSE = 4.39px**；RANSAC 内点重投影 RMSE = 1.62px。
- RANSAC 内点 18 个；重投影误差 ≤3px 的点 17/36；最大单点误差 8.77px。
- 标定参考距离 Z0 ≈ 0.6m（记录值，仅标注该 H 对应的工作距离，不参与求解）。
- 验收（RMSE ≤ 5px 且 ≥4 组 且 ≥30 点）：**通过**。

逐组重投影误差（基于最终 H）：

| 组 | 帧对 | 点数 | RMSE(px) | 最大误差(px) | ≤3px 点数 |
|----|------|------|----------|--------------|-----------|
| 0  | pair_00 | 9 | 3.84 | 6.14 | 4 |
| 1  | pair_01 | 9 | 5.13 | 8.74 | 3 |
| 2  | pair_02 | 9 | 3.83 | 8.68 | 6 |
| 3  | pair_03 | 9 | 4.63 | 8.77 | 4 |
| 合计 | — | 36 | 4.39 | 8.77 | 17 |

完整逐点数据见 `results/reprojection_errors.csv`，单应矩阵数值见 `results/homography_matrix.txt` 与 `results/homography.npy`。

## 8. 重要讨论：单应在近距离下的深度依赖（视差）

单应矩阵在两相机之间**仅对同一平面严格成立**。本系统基线约 4cm，在 ~0.6m 近距离下视差不可忽略：
当标定板处于不同深度时，对应点落在不同平面上，单一全局单应无法同时精确拟合。

实验佐证（标定调试阶段）：每一组**单独**用 9 点求解时自洽 RMSE 仅 0.8~2.3px，证明人工对应点准确；
但把一组明显处于不同深度的数据并入全局求解时，该组重投影误差升至约 18px，使总体 RMSE 显著恶化。
据此结论：**应将所有标定位置固定在同一工作距离**（变化画面内位置而非深度），最终数据即在近似一致深度下采集，
全局 RMSE 收敛到 4.39px 并通过验收。该现象也说明：在更远工作距离下，4cm 基线引入的视差更小，
单应的精度与深度容差都会更好。

## 9. 运行期使用（接口）

```python
from registration import load_homography, project_point, project_box
from types import DetectionBox

H = load_homography()                         # 加载 config/homography.npy
xy_rgb = project_point((x_ir, y_ir), H)       # IR 点 -> RGB 点
box_rgb = project_box(DetectionBox(x1,y1,x2,y2), H)  # IR 框 -> RGB 轴对齐外接框
# box_rgb.quad 为 4 个变换后角点（精确四边形）；box_rgb 本身为外接框，便于 IoU 关联
```

注意：单应把矩形映射成一般四边形，`project_box` 主返回 4 角变换后的**轴对齐外接框**，并附精确四边形角点。

## 10. 文件清单

```
results/
  homography.npy           最终单应矩阵 (3x3 float64, IR->RGB)
  homography_matrix.txt    同上的可读文本
  homography_meta.json     元数据(日期/组数/点数/Z0/RMSE/内外点/分辨率等)
  reprojection_errors.csv  逐点原始误差(含 ir/rgb/投影坐标、误差、内/外点标记)
  per_group_summary.csv    逐组误差汇总
data/
  correspondences.json     全部对应点原始记录(每组9对点+帧对名+时间戳)
  overlay.png              误差可视化(RGB上叠加: 绿=点选RGB点, 红/橙=IR投影点, 连线+误差数)
  pair_0X_rgb.png          各组可见光原图(640x480)
  pair_0X_ir.png           各组红外可视化原图(256x192, 已roll对齐)
  pair_0X_ir_temp.npy      各组红外原始温度矩阵(256x192 uint16, 备查)
code/
  calibrate_homography.py  交互式标定工具(网页端: 采集+点选+求解+误差报告)
  registration.py          运行期配准(load_homography/project_point/project_box)
  types.py                 DetectionBox 数据类型
  README_calibration.md    标定工具使用说明
```

## 11. 复现要点

- 标定取流：RGB 640×480@60(MJPG, GStreamer)，IR 经 uvc_demo 输出 temp+yuv、套用固定 roll 对齐。
- 关键参数：同步阈值 33ms；RANSAC 重投影阈值 3.0px；每组 9 点、固定左→右上→下顺序。
- 验收阈值：总体 RMSE ≤ 5px、组数 ≥4、汇总点数 ≥30。
