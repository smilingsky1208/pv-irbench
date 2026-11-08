# -*- coding: utf-8 -*-
"""
PV-IRBench · FAR 与图像级召回评测脚本 v0.1

设计原则：**指标计算与训练框架解耦**。
  - compute_metrics() 是纯 Python 实现，输入 (真值标签, 检出框数) 即可算指标，
    不依赖任何深度学习框架 —— 因此可评测任何来源的模型（YOLO / RT-DETR / 云端 API / 人工判读）。
  - run_yolo() 只是一个可选适配器，方便快速上手。

许可：AGPL-3.0-or-later（见 LICENSE）

用法：
  # 单阈值评测（自有数据集，目录结构需含 images/ 与 labels/）
  python eval_far.py --source own --data D:/data/solar_panel_v3_clean --split valid --conf 0.10

  # 扫描多阈值，生成 FAR-召回权衡曲线
  python eval_far.py --source own --data D:/data/solar_panel_v3_clean --sweep

  # 只算指标（已有预测结果 JSON：[{"gt": "...", "n_boxes": 2}, ...]）
  python eval_far.py --from-json predictions.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

NEG = "No-Anomaly"
DEFAULT_SWEEP = [0.25, 0.10, 0.05, 0.01]


# ----------------------------------------------------------------------------
# 核心：指标计算（纯 Python，与框架无关）
# ----------------------------------------------------------------------------
def compute_metrics(records):
    """
    records: 迭代对象，元素为 dict，含
        - "gt":      真值标签（字符串；"No-Anomaly" 表示负样本）
        - "n_boxes": 该图检出的框数（int）
    返回指标字典。
    """
    neg_total = neg_fp = neg_fp_boxes = 0
    per_cls = defaultdict(lambda: {"total": 0, "hit": 0, "miss": 0})

    for r in records:
        gt, nb = r["gt"], int(r["n_boxes"])
        if gt == NEG:
            neg_total += 1
            if nb > 0:
                neg_fp += 1
                neg_fp_boxes += nb
        else:
            s = per_cls[gt]
            s["total"] += 1
            if nb > 0:
                s["hit"] += 1
            else:
                s["miss"] += 1

    far = neg_fp / neg_total if neg_total else 0.0
    fp_per_1k = neg_fp_boxes / neg_total * 1000 if neg_total else 0.0
    tp = sum(v["hit"] for v in per_cls.values())
    fn = sum(v["miss"] for v in per_cls.values())
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "FAR": round(far, 4),
        "fp_boxes_per_1000": round(fp_per_1k, 1),
        "image_level_recall": round(recall, 4),
        "negative": {"total": neg_total, "fp": neg_fp, "tn": neg_total - neg_fp},
        "positive": {"total": tp + fn, "hit": tp, "miss": fn},
        "per_class": {
            k: dict(v, recall=round(v["hit"] / v["total"], 4) if v["total"] else 0.0)
            for k, v in per_cls.items()
        },
    }


def print_report(m, conf=None, n=None, title="评测结果"):
    print("=" * 62)
    print(f"{title} · 图像级")
    print("=" * 62)
    if conf is not None:
        print(f"conf={conf}" + (f"    样本数={n}" if n else ""))
        print("-" * 62)
        print(f"【FAR 假警报率】      {m['FAR'] * 100:.2f}%   "
              f"({m['negative']['fp']}/{m['negative']['total']} 张正常组件被误报)")
        print(f"【每千块误报框数】    {m['fp_boxes_per_1000']:.1f} 个/1000 块")
        print(f"【图像级召回】        {m['image_level_recall'] * 100:.2f}%   "
              f"({m['positive']['hit']}/{m['positive']['total']})")
        print("-" * 62)
        print(f"{'类别':<18}{'样本':>6}{'命中':>6}{'漏检':>6}{'召回':>9}")
        for cls in sorted(m["per_class"], key=lambda k: -m["per_class"][k]["total"]):
            v = m["per_class"][cls]
            print(f"{cls:<18}{v['total']:>6}{v['hit']:>6}{v['miss']:>6}{v['recall'] * 100:>8.1f}%")
    print("=" * 62)


# ----------------------------------------------------------------------------
# 数据加载：自有 YOLO 格式数据集
# ----------------------------------------------------------------------------
def load_own(base: Path, split: str, names=None):
    """负样本 = 无标注框的图；正样本 = 有框的图，标签取出现最多的类别。"""
    ip, lp = base / split / "images", base / split / "labels"
    if not ip.exists():
        ip, lp = base / "images" / split, base / "labels" / split
    items = []
    for im in sorted(list(ip.glob("*.jpg")) + list(ip.glob("*.png"))):
        lf = lp / (im.stem + ".txt")
        ids = []
        if lf.exists():
            for line in lf.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    ids.append(int(line.split()[0]))
        if not ids:
            items.append((str(im), NEG))
        else:
            main = max(set(ids), key=ids.count)
            items.append((str(im), names.get(main, str(main)) if names else str(main)))
    return items


# ----------------------------------------------------------------------------
# 可选适配器：Ultralytics YOLO
# ----------------------------------------------------------------------------
def run_yolo(weights, paths, conf, batch=32, names=None):
    from ultralytics import YOLO  # AGPL-3.0，仅在需要推理时导入
    model = YOLO(weights)
    if names is None:
        names = model.names
    n_boxes = []
    for i in range(0, len(paths), batch):
        for r in model(paths[i:i + batch], conf=conf, verbose=False):
            n_boxes.append(0 if r.boxes is None else len(r.boxes))
    return n_boxes, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="own", choices=["own"])
    ap.add_argument("--data", help="数据集根目录（含 images/ 与 labels/）")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--weights", help="模型权重；仅 --source own 推理时需要")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--sweep", action="store_true", help="扫描多个阈值生成权衡曲线")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--from-json", help="直接读取预测结果 JSON 计算指标")
    ap.add_argument("--out", default="results/eval_result.json")
    args = ap.parse_args()

    # 模式 A：已有预测结果，只算指标
    if args.from_json:
        m = compute_metrics(json.load(open(args.from_json, encoding="utf-8")))
        print_report(m, n=len(json.load(open(args.from_json, encoding="utf-8"))),
                     title="指标计算（来自预测结果）")
        return

    base = Path(args.data)
    if args.weights:
        from ultralytics import YOLO
        names = YOLO(args.weights).names
    else:
        names = None
    samples = load_own(base, args.split, names)
    print(f"样本: {len(samples)} 张（{base.name}/{args.split}）")
    neg = sum(1 for _, g in samples if g == NEG)
    print(f"其中负样本 {neg} 张（{neg / max(len(samples), 1) * 100:.1f}%）")
    if neg == 0:
        print("⚠️ 警告：该数据集没有负样本，FAR 无法测量。")

    paths = [p for p, _ in samples]
    gts = [g for _, g in samples]

    if args.sweep:
        curve = []
        for c in DEFAULT_SWEEP:
            nb, _ = run_yolo(args.weights, paths, c, args.batch, names)
            m = compute_metrics([{"gt": g, "n_boxes": n} for g, n in zip(gts, nb)])
            m["conf"] = c
            curve.append(m)
            print(f"\n--- conf={c} ---")
            print_report(m, conf=c, n=len(samples))
        print(f"\n{'conf':<8}{'FAR':>10}{'每千块误报':>12}{'图像级召回':>12}")
        for m in curve:
            print(f"{m['conf']:<8}{m['FAR'] * 100:>9.2f}%"
                  f"{m['fp_boxes_per_1000']:>12.1f}{m['image_level_recall'] * 100:>11.2f}%")
        out_data = {"curve": curve}
    else:
        nb, _ = run_yolo(args.weights, paths, args.conf, args.batch, names)
        m = compute_metrics([{"gt": g, "n_boxes": n} for g, n in zip(gts, nb)])
        print_report(m, conf=args.conf, n=len(samples))
        out_data = m
        out_data["conf"] = args.conf

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_data["dataset"] = f"{base.name}/{args.split}"
    out_data["n_samples"] = len(samples)
    out.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
