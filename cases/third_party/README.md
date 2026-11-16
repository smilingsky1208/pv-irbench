# 第三方模型同尺复算案例

本报告对应公开文章《mAP 78% 的模型，误报率 54%：我们把四款光伏 AI 检测模型放在了同一把尺子下》。

本目录公开的是**可复核性材料**：原始数据、验证集清单、评测脚本、两版报告。不包含任何模型权重或原始图像。

## 文件

| 文件 | 说明 |
|---|---|
| `第三方复算报告_简洁版.pdf` | 1 页决策摘要 |
| `第三方复算报告_完整版.pdf` | 3 页技术报告，含全阈值数据与边界声明 |
| `batch_result.json` | 四模型 × 7 阈值原始结果，含 FAR、误报框/千块、图像级召回 |
| `valid_manifest.csv` | 503 张验证集的图像名与真值标签（151 张为纯负样本） |
| `recompute.py` | 公开版复算脚本，调用 `../../benchmark/eval_far.py` 的 `compute_metrics()` |
| `gen_reports.py` | 报告生成器（`reportlab`，中文字体为系统微软雅黑） |

## 复现方式

### 1. 准备验证集

`solar_panel_v3_clean/valid/` 为本项目的自建验证集，由公开数据清洗、去重并重新划分而来。本目录提供 `valid_manifest.csv`，列出全部 503 张图像的文件名与真值，可用于核对划分一致性。

原始图像来源：[Raptor Maps - InfraredSolarModules](https://github.com/raptormaps/InfraredSolarModules)（MIT 许可，20,000 张航拍红外图）。本项目的清洗与划分脚本未随仓库发布；验证集划分属于研究资产，可来信索取用于学术复核。

### 2. 准备权重

| 模型 | 仓库 | 权重路径 | 大小 | 许可 |
|---|---|---|---|---|
| SolarPanelsDefectDetector | P-Darabi/SolarPanelsDefectDetector | `ThermalDetector/Th_G_v9.pt` | 51.6 MB | MIT |
| SolarVulture | Ramykaz/Solarvulture_oasis | `Model/anomaly_detection.pt` | 22.5 MB | **未声明** |
| Skytroll | Skytroll914/SolarPanelDefectDetection_YOLO | `best.pt` | 5.4 MB | **未声明**，可见光 RGB |

**自研基准权重不公开。** 脚本在权重缺失时自动跳过该项，第三方模型结果仍可完整复算；自研模型的全部指标在 `batch_result.json` 中保留。

### 3. 运行

```bash
export PVIRBENCH_DATA_ROOT=/path/to/data/root
export PVIRBENCH_WEIGHTS=/path/to/weights
python recompute.py
```

其中 `DATA_ROOT/solar_panel_v3_clean/valid/images/` 与 `labels/` 需按上文目录结构就位。运行环境：`ultralytics 8.4.120`，`batch=32`。

## 口径重申

- FAR：负样本中出现任一检出框即算一次假警报。
- 图像级召回：一张图检出任一框即算命中，不代表框级定位精度。
- 真值固定为数据集自身 4 类，不使用被测模型的类别名。
- 同域验证，不代表真实电站表现。模态差异可能对外来模型不利。
- 本报告为算法评测结果，不具备 CMA/CNAS 资质。

评测方：张笑天 · 湖南大学机器人视觉感知与控制技术国家工程研究中心
报告编号：PV-IRBench-3RD-20261116
