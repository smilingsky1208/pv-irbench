# -*- coding: utf-8 -*-
"""
PV-IRBench · 多模型同尺复算（公开版）

对同一份验证集、同一批阈值，评测多个第三方模型 + 自研基准。
真值固定使用数据集自身 4 类标注，不受被测模型类别名影响。
指标算法直接调用 ../../benchmark/eval_far.py 的 compute_metrics()，未作修改。

用法：
    export PVIRBENCH_DATA_ROOT=/path/to/your/data/root
    export PVIRBENCH_WEIGHTS=/path/to/your/weights
    python recompute.py

目录约定（DATA_ROOT 下）：
    solar_panel_v3_clean/valid/images/*.jpg
    solar_panel_v3_clean/valid/labels/*.txt

权重获取（第三方模型，均可公开下载）：
    YOLOv9  Th_G_v9.pt
        https://github.com/P-Darabi/SolarPanelsDefectDetector
        路径 ThermalDetector/Th_G_v9.pt，51.6 MB，MIT 许可
    SolarVulture  anomaly_detection.pt
        https://github.com/Ramykaz/Solarvulture_oasis
        路径 Model/anomaly_detection.pt，22.5 MB，许可未声明
    Skytroll  best.pt
        https://github.com/Skytroll914/SolarPanelDefectDetection_YOLO
        路径 best.pt，5.4 MB，许可未声明，可见光 RGB（跨模态对照）

自研基准权重为私有资产，不随仓库发布。缺失时脚本自动跳过该项，
其余第三方结果仍可完整复算；原始全量结果见 batch_result.json。
"""
import os
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                      # pv-irbench/
sys.path.insert(0, str(REPO / "benchmark"))

from eval_far import compute_metrics, NEG  # noqa: E402
from ultralytics import YOLO               # noqa: E402

DATA_ROOT = Path(os.environ.get(
    "PVIRBENCH_DATA_ROOT", str(HERE.parents[2])
)).resolve()
WEIGHTS = Path(os.environ.get(
    "PVIRBENCH_WEIGHTS", str(DATA_ROOT / "opensource_model")
)).resolve()

DATA = DATA_ROOT / "solar_panel_v3_clean" / "valid"
NAMES = ["ByPassed", "Diode", "HotSpot", "StringFault"]
SWEEP = [0.50, 0.40, 0.30, 0.25, 0.10, 0.05, 0.01]
OUT = HERE / "batch_result.json"

MODELS = [
    {
        "id": "self_v8", "label": "自研 YOLOv8（基准）",
        "weights": DATA_ROOT / "competition/runs/final/weights/best.pt",
        "source": "本项目自研（权重不公开）", "license": "私有（对照基准）",
        "modality": "红外", "note": "4 类，同域验证集训练",
    },
    {
        "id": "pd_v9", "label": "SolarPanelsDefectDetector YOLOv9",
        "weights": WEIGHTS / "Th_G_v9.pt",
        "source": "github.com/P-Darabi/SolarPanelsDefectDetector",
        "license": "MIT（可商用，需署名）", "modality": "红外灰度",
        "note": "自报 mAP50 78%；README 称 8 类，权重实际 9 类",
    },
    {
        "id": "solarvulture", "label": "SolarVulture anomaly_detection",
        "weights": WEIGHTS / "solarvulture_anomaly.pt",
        "source": "github.com/Ramykaz/Solarvulture_oasis",
        "license": "未声明（GitHub license=null）", "modality": "红外",
        "note": "5 类；作者注明类别不均衡、部分类样本 <20",
    },
    {
        "id": "skytroll", "label": "SolarPanelDefectDetection YOLO26",
        "weights": WEIGHTS / "skytroll_best.pt",
        "source": "github.com/Skytroll914/SolarPanelDefectDetection_YOLO",
        "license": "未声明（GitHub license=null）", "modality": "可见光 RGB",
        "note": "5 类含显式 Non Defective；跨模态对照组",
    },
]


def load_samples():
    """读取验证集，真值取该图出现频次最高的类别。"""
    ip, lp = DATA / "images", DATA / "labels"
    items = []
    for im in sorted(list(ip.glob("*.jpg")) + list(ip.glob("*.png"))):
        lf = lp / (im.stem + ".txt")
        ids = []
        if lf.exists():
            for line in lf.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    ids.append(int(line.split()[0]))
        gt = NEG if not ids else NAMES[max(set(ids), key=ids.count)]
        items.append((str(im), gt))
    return items


def infer(model, paths, conf, batch=32):
    nb = []
    for i in range(0, len(paths), batch):
        for r in model(paths[i:i + batch], conf=conf, verbose=False):
            nb.append(0 if r.boxes is None else len(r.boxes))
    return nb


def main():
    if not DATA.exists():
        raise SystemExit(f"验证集不存在：{DATA}。请设置 PVIRBENCH_DATA_ROOT。")

    samples = load_samples()
    paths = [p for p, _ in samples]
    gts = [g for _, g in samples]
    neg = sum(1 for g in gts if g == NEG)
    print(f"数据根目录: {DATA_ROOT}")
    print(f"样本 {len(samples)} 张，负样本 {neg} 张（{neg/len(samples)*100:.1f}%）")
    print(f"阈值 {SWEEP}\n", flush=True)

    result = {
        "protocol": "PV-IRBench v0.1",
        "dataset": "solar_panel_v3_clean/valid",
        "n_samples": len(samples), "n_negative": neg,
        "n_positive": len(samples) - neg,
        "sweep": SWEEP, "models": {},
    }

    for spec in MODELS:
        wp = Path(spec["weights"])
        if not wp.exists():
            print(f"[跳过] {spec['label']}：权重不存在 {wp}", flush=True)
            continue
        print(f"===== {spec['label']} =====", flush=True)
        model = YOLO(str(wp))
        curve = []
        for conf in SWEEP:
            nb = infer(model, paths, conf)
            m = compute_metrics([{"gt": g, "n_boxes": n}
                                 for g, n in zip(gts, nb)])
            m["conf"] = conf
            curve.append(m)
            print(f"  conf={conf:<5} FAR={m['FAR']*100:>6.2f}%  "
                  f"误报框/千块={m['fp_boxes_per_1000']:>7.1f}  "
                  f"召回={m['image_level_recall']*100:>6.2f}%", flush=True)
        entry = dict(spec)
        entry["weights"] = str(wp)
        entry["n_classes"] = len(model.names)
        entry["class_names"] = model.names
        entry["curve"] = curve
        zero_far = [c for c in curve if c["FAR"] == 0.0]
        entry["best_recall_at_zero_far"] = (
            max(zero_far, key=lambda c: c["image_level_recall"])
            if zero_far else None
        )
        result["models"][spec["id"]] = entry
        print(flush=True)

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"已保存: {OUT}")

    print("\n" + "=" * 92)
    print("同尺对比摘要（FAR / 图像级召回）")
    print("=" * 92)
    print(f"{'conf':<7}" + "".join(
        f"{m['label'][:14]:>18}" for m in MODELS
        if m["id"] in result["models"]))
    for i, conf in enumerate(SWEEP):
        row = f"{conf:<7}"
        for spec in MODELS:
            if spec["id"] not in result["models"]:
                continue
            c = result["models"][spec["id"]]["curve"][i]
            row += f"{c['FAR']*100:>8.2f}%/{c['image_level_recall']*100:>7.2f}%"
        print(row)
    print("\nFAR=0 最高召回：")
    for spec in MODELS:
        if spec["id"] not in result["models"]:
            continue
        b = result["models"][spec["id"]]["best_recall_at_zero_far"]
        summary = ("%.2f%% @ conf=%s" % (b["image_level_recall"] * 100, b["conf"])
                   if b else "无 FAR=0 工作点")
        print("  %-26s %s" % (spec["label"][:24], summary))


if __name__ == "__main__":
    main()
