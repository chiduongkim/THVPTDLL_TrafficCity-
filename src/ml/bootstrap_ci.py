"""
Khoang tin cay bootstrap theo KHOI NGAY cho metrics tren tap test.

Cac dong lien tiep cua cung sensor/ngay tuong quan manh (mot dot un tac keo dai nhieu
buoc), nen lay mau lai theo ngay lich thay vi theo dong. Tap test chi ~9-10 ngay, khoang
tin cay vi vay rong; day la muc do khong chac chan that cua ket qua.

Chay:  python src/ml/bootstrap_ci.py --resolution 15min
Ghi:   reports/congestion/pems08_<res>/bootstrap_ci_test.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_IDS, CLASS_NAMES, RESOLUTIONS, ROOT, load_config  # noqa: E402


def metrics_from_cm(cm):
    cm = cm.astype("float64")
    sup = cm.sum(1)
    pred = cm.sum(0)
    tp = np.diag(cm)
    rec = np.divide(tp, sup, out=np.zeros(3), where=sup > 0)
    prec = np.divide(tp, pred, out=np.zeros(3), where=pred > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(3), where=(prec + rec) > 0)
    present = sup > 0
    return {"macro_f1_present_classes": f1[present].mean(), "balanced_accuracy": rec[present].mean(),
            "f1_congested": f1[1], "f1_severe": f1[2], "recall_severe": rec[2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", choices=list(RESOLUTIONS), default="15min")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config()
    res, seed = args.resolution, cfg["seed"]
    pr = pd.read_parquet(ROOT / cfg["paths"]["processed_dir"] / f"predictions_{res}_test.parquet")
    pr["day"] = pr["feature_timestamp"].dt.date
    days = sorted(pr["day"].unique())
    models = list(pr["model_name"].unique())

    # Ma tran nham lan theo (mo hinh, ngay)
    cms = {}
    for m in models:
        d = pr[pr["model_name"] == m]
        arr = np.zeros((len(days), 3, 3), dtype="int64")
        for i, day in enumerate(days):
            x = d[d["day"] == day]
            np.add.at(arr[i], (x["actual_level"].values.astype(int), x["predicted_level"].values.astype(int)), 1)
        cms[m] = arr

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), size=(args.n_boot, len(days)))
    boot = {m: [metrics_from_cm(cms[m][i].sum(0)) for i in idx] for m in models}
    point = {m: metrics_from_cm(cms[m].sum(0)) for m in models}

    out = {"resolution": res, "n_boot": args.n_boot, "seed": seed, "block": "ngay lich cua feature_timestamp",
           "n_blocks": len(days), "days": [str(d) for d in days],
           "test_class_support": {CLASS_NAMES[i]: int(cms[models[0]].sum((0, 2))[i]) for i in CLASS_IDS},
           "models": {}, "paired_differences": {}}
    keys = list(point[models[0]])
    for m in models:
        out["models"][m] = {k: {"estimate": float(point[m][k]),
                                "ci95": [float(np.percentile([b[k] for b in boot[m]], 2.5)),
                                         float(np.percentile([b[k] for b in boot[m]], 97.5))]} for k in keys}
    for a, b in (("random_forest", "baseline_persistence"), ("decision_tree", "baseline_persistence"),
                 ("random_forest", "decision_tree")):
        out["paired_differences"][f"{a} - {b}"] = {}
        for k in ("macro_f1_present_classes", "balanced_accuracy", "f1_severe"):
            diff = np.array([x[k] - y[k] for x, y in zip(boot[a], boot[b])])
            out["paired_differences"][f"{a} - {b}"][k] = {
                "estimate": float(point[a][k] - point[b][k]),
                "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                "share_resamples_a_better": float((diff > 0).mean())}
    path = ROOT / cfg["paths"]["reports_dir"] / f"pems08_{res}" / "bootstrap_ci_test.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[{res}] bootstrap theo {len(days)} ngay, {args.n_boot} lan -> {path}")
    for m in models:
        e = out["models"][m]["macro_f1_present_classes"]
        b = out["models"][m]["balanced_accuracy"]
        print(f"  {m:22s} macro-F1 {e['estimate']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]   "
              f"bal-acc {b['estimate']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]")
    for k, v in out["paired_differences"].items():
        e = v["macro_f1_present_classes"]
        print(f"  {k:46s} dMacroF1 {e['estimate']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}] "
              f"P(a>b)={e['share_resamples_a_better']:.2f}")


if __name__ == "__main__":
    main()
