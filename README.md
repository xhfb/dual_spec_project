# dual_spec_project

RDK X5 双光谱协同感知系统运行代码。

**GitHub 真源**：[https://github.com/xhfb/dual_spec_project](https://github.com/xhfb/dual_spec_project)

**板端路径**：`/home/sunrise/dual_spec_project`

## 同步

```bash
git clone https://github.com/xhfb/dual_spec_project.git
cd dual_spec_project
git pull origin main
```

本目录为知识库内 clone，与 GitHub 保持同步：

```bash
cd 04-software/dual_spec_project
git pull
```

## 主 Demo

```bash
python3 scripts/dual_fusion_web.py --ptz
# 浏览器 http://<板子IP>:8004/
```

## 论文评测（知识库扩展，需 push 回 GitHub）

| 脚本 | 用途 |
|------|------|
| `scripts/eval_nine_groups.py` | T9 九组实验 |
| `scripts/eval_ablation.py` | 消融 A1/A2/A3 |
| `scripts/generate_paper_figures.py` | 图 5-1 / 表 5-1 |
| `scripts/exp_*.py` | 专项实验 |
| `src/experiments/pipeline.py` | 共享评测流水线 |

详见知识库 [`../../06-testing/DATA_COLLECTION_RUNBOOK.md`](../../06-testing/DATA_COLLECTION_RUNBOOK.md)。

## 模块

- `src/detection/ir_thermal.py` — IR 热检测
- `src/fusion/` — 配准 + Late Fusion
- `src/tracking/` — IoU + EMA 跟踪
- `src/ptz/` — PD 云台
- `config/` — homography + yaml 参数

归档说明：[`README_ARCHIVE.md`](README_ARCHIVE.md)
