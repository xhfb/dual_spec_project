# dual_spec_project 归档索引

> 来源：[GitHub xhfb/dual_spec_project](https://github.com/xhfb/dual_spec_project)  
> 板端路径：`/home/sunrise/dual_spec_project`  
> 日期：2026-06-05

## 系统闭环

RGB 640×480 → YOLOv11 人体检测 ──┐  
IR 256×192 → 温度矩阵热检测 ──────┼→ Late Fusion（IoU + 场景加权）  
                                  ↓  
                         单应 H：IR→RGB 投影（RMSE≈2.48 px）  
                                  ↓  
                         主目标选择 + IoU 多帧跟踪  
                                  ↓  
                         PD 云台控制 + 10s 无目标回中

## 入口

| 脚本 | 端口 |
|------|------|
| `scripts/dual_fusion_web.py` | 8004 主系统 |
| `scripts/person_detect_web.py` | 8002 RGB |
| `scripts/ir_thermal_detect_web.py` | 8003 IR 调试 |
| `scripts/calibrate_homography.py` | 8001 标定 |

## 配置

| 文件 | 说明 |
|------|------|
| `config/homography.npy` | IR→RGB 单应 |
| `config/fusion.yaml` | 融合权重 |
| `config/ir_thermal.yaml` | IR 检测 |
| `config/tracking.yaml` | 跟踪 |
| `config/ptz.yaml` | 云台 PD |

标定归档亦见 `dist/ir_rgb_calibration_2026-06-05/` 与知识库 `06-testing/calibration-data/`。

## 论文 API

```python
from src.detection.ir_thermal import IRThermalDetector
from src.fusion.late_fusion import fuse_detections, load_fusion_config
from src.fusion.registration import load_homography
from src.tracking.tracker import TargetTracker, load_tracking_config
```

## 关联文档

- 知识库采数手册：[`../../06-testing/DATA_COLLECTION_RUNBOOK.md`](../../06-testing/DATA_COLLECTION_RUNBOOK.md)
- Agent 提示词：[`../prompts/`](../prompts/)
