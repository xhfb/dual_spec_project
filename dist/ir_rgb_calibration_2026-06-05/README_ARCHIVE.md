# 归档包快速索引（给下游 Agent）

## 这是什么

`dual_spec_project` 红外(IR) → 可见光(RGB) **单应标定**的一次完整交付物，日期 **2026-06-05**。
用于：数据归档、论文方法/实验章节撰写、结果复现核对。

## 核心结论（可直接引用）

- **方法**：人工点选 IR 热斑 ↔ RGB 红点对应，RANSAC 求 3×3 单应矩阵 H。
- **规模**：6 组、50 个对应点（部分组未满 9 点，因视野遮挡）。
- **精度**：总体重投影 RMSE = **2.48 px**；内点 RMSE = **1.73 px**。
- **工作距离**：Z0 = **1.5 m**（元数据标注，H 求解不依赖距离）。
- **验收**：`homography_meta.json` 中 `passed: true`。

## 优先阅读顺序

1. `README_for_paper.md` — 完整方法学 + 结果表 + 文件说明（论文主文档）
2. `results/homography_meta.json` — 结构化指标摘要
3. `results/per_group_summary.csv` — 逐组 RMSE
4. `results/reprojection_errors.csv` — 50 行逐点误差（画散点图/箱线图用）
5. `data/overlay.png` — 重投影误差可视化图（论文插图候选）
6. `data/correspondences.json` — 原始人工标注坐标（审计/复算用）

## 关键文件路径

| 用途 | 路径 |
|------|------|
| 最终 H 矩阵 | `results/homography.npy` |
| H 可读文本 | `results/homography_matrix.txt` |
| 元数据 | `results/homography_meta.json` |
| 原始对应点 | `data/correspondences.json` |
| 误差可视化 | `data/overlay.png` |
| 同步帧对图像 | `data/pair_*_rgb.png`, `data/pair_*_ir.png` |
| IR 温度原始矩阵 | `data/pair_*_ir_temp.npy` |

## 坐标系提醒

- IR 输入坐标：**256×192**，已 roll 对齐（`roll_x=-164, roll_y=-2`）。
- RGB 输出坐标：**640×480**。
- 运行期加载 H 后，用 `code/registration.py` 的 `project_point` / `project_box` 做投影。

## 数据异常说明

- `pair_02` 图像存在于 `data/`，但**未纳入** `correspondences.json`（该位置被跳过）。
- 组 2（`pair_03`）仅 6 个对应点；组 4（`pair_05`）为 8 个对应点。
- `correspondences.json` 中 `num_points_per_group: 9` 为工具默认值，实际各组点数以上表为准。

## 打包来源

```
源目录:
  dual_spec_project/calib_data/   -> data/
  dual_spec_project/config/       -> results/ (homography.npy + homography_meta.json)
衍生生成:
  results/homography_matrix.txt
  results/reprojection_errors.csv
  results/per_group_summary.csv
```
