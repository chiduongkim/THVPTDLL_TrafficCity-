"""
Huan luyen va danh gia mo hinh du doan muc un tac tai t + 1 gio (benchmark PEMS08).

Quy trinh:
  1. Doc bo du lieu da gan nhan (build_pems08_dataset.py).
  2. Tao dac trung CHI tu thong tin <= t (gia tri hien tai, do tre qua khu, cua so truot
     lui, gio/ngay trong tuan). Khong dung cot tuong lai, khong co thong ke toan cuc.
  3. Chia theo MOC THOI GIAN 70/15/15 (tat ca sensor cung timestamp nam cung mot tap),
     co khoang cach (embargo) bang horizon o cuoi train va validation de nhan khong
     chui sang tap sau.
  4. Chon tham so Decision Tree / Random Forest tren validation (macro-F1).
  5. Danh gia test DUNG MOT LAN cho mo hinh da chon va hai baseline.
  6. Luu model, metrics, du doan test, bieu do.

Chay:  python src/ml/train_congestion_models.py --resolution 15min
"""
import argparse
import itertools
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASS_IDS, CLASS_NAMES, RESOLUTIONS, ROOT, horizon_steps,  # noqa: E402
                    load_config)

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# Lag (so buoc) va cua so truot (so buoc ~ 1 gio) theo do phan giai
FEATURE_SPEC = {
    "15min": {"lags": [1, 2, 3, 4, 8], "roll": 4},
    "5min": {"lags": [1, 2, 3, 6, 12, 24], "roll": 12},
}
DT_GRID = {"max_depth": [6, 10, 14], "min_samples_leaf": [20, 100],
           "class_weight": [None, "balanced"]}
RF_GRID = {"max_depth": [12, 20], "min_samples_leaf": [20, 100],
           "class_weight": [None, "balanced_subsample"]}
RF_FIXED = {"n_estimators": 100, "max_features": "sqrt", "bootstrap": True}
RF_MAX_SAMPLES = {"15min": 0.3, "5min": 0.15}


# ---------------------------------------------------------------- dac trung
def build_features(df, res):
    """Tra ve (X, valid_mask, info). Chi dung gia tri tai t va qua khu cua cung sensor."""
    spec = FEATURE_SPEC[res]
    step = RESOLUTIONS[res]["step_minutes"]
    g = df.groupby("sensor_id", sort=False)

    # Kiem tra luoi thoi gian lien tuc de dich theo vi tri la dung
    lmax = max(spec["lags"])
    lag_ts = g["timestamp"].shift(lmax)
    ok = lag_ts.notna()
    if not ((df.loc[ok, "timestamp"] - lag_ts[ok]) == pd.Timedelta(minutes=lmax * step)).all():
        raise ValueError("Luoi thoi gian khong lien tuc, khong the dich lag theo vi tri")

    X = pd.DataFrame(index=df.index)
    for c in ("speed_kmh", "flow", "occupancy", "zero_flow_steps"):
        X[c] = df[c].astype("float32")
    for lag in spec["lags"]:
        for c in ("speed_kmh", "flow", "occupancy"):
            X[f"{c}_lag{lag * step}m"] = g[c].shift(lag).astype("float32")
    w = spec["roll"]
    roll = g["speed_kmh"].rolling(w, min_periods=w)
    X["speed_roll_mean_1h"] = roll.mean().reset_index(level=0, drop=True).astype("float32")
    X["speed_roll_std_1h"] = roll.std().reset_index(level=0, drop=True).astype("float32")
    X["speed_roll_min_1h"] = roll.min().reset_index(level=0, drop=True).astype("float32")
    X["flow_roll_mean_1h"] = g["flow"].rolling(w, min_periods=w).mean().reset_index(level=0, drop=True).astype("float32")
    X["speed_delta_1h"] = (X["speed_kmh"] - X[f"speed_kmh_lag{60}m"]).astype("float32")
    ts = df["timestamp"]
    X["time_of_day_h"] = (ts.dt.hour + ts.dt.minute / 60).astype("float32")
    X["day_of_week"] = ts.dt.dayofweek.astype("float32")
    X["is_weekend"] = (ts.dt.dayofweek >= 5).astype("float32")

    valid = X.notna().all(axis=1)
    return X, valid.values, {"lags_steps": spec["lags"], "roll_steps": w, "step_minutes": step}


# --------------------------------------------------------------------- chia
def time_split(ts, cfg, res):
    """Chia theo moc timestamp. Tra ve mask train/val/test (da bo embargo) va thong tin moc."""
    s = cfg["split"]
    if abs(s["train"] + s["validation"] + s["test"] - 1.0) > 1e-9:
        raise ValueError("Ty le chia khong cong lai bang 1")
    uniq = np.sort(ts.unique())
    n = len(uniq)
    i1 = int(round(n * s["train"]))
    i2 = int(round(n * (s["train"] + s["validation"])))
    val_start, test_start = uniq[i1], uniq[i2]
    step = RESOLUTIONS[res]["step_minutes"]
    hz = np.timedelta64(cfg["horizon_minutes"], "m")
    t = ts.values
    train = t < (val_start - hz)          # embargo: nhan t+1h phai nam truoc val
    val = (t >= val_start) & (t < (test_start - hz))
    test = t >= test_start
    info = {
        "unique_timestamps_usable": int(n),
        "train_range": [str(uniq[0]), str(val_start - hz - np.timedelta64(step, "m"))],
        "validation_range": [str(val_start), str(test_start - hz - np.timedelta64(step, "m"))],
        "test_range": [str(test_start), str(uniq[-1])],
        "embargo_minutes": int(cfg["horizon_minutes"]),
        "note": "Hang cuoi cua train/validation (trong khoang horizon) bi bo de nhan khong lay du lieu tap sau.",
    }
    return train, val, test, info


# ---------------------------------------------------------------- danh gia
def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_IDS, zero_division=0)
    present = sup > 0
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1_present_classes": float(f[present].mean()),
        "macro_f1_all_3_classes": float(f.mean()),
        "per_class": {CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                       "f1": float(f[i]), "support": int(sup[i])} for i in CLASS_IDS},
        "confusion_matrix": cm.tolist(),
        "predicted_counts": {CLASS_NAMES[i]: int((y_pred == i).sum()) for i in CLASS_IDS},
    }


def proba_full(model, X):
    pr = model.predict_proba(X)
    out = np.zeros((len(X), 3), dtype="float32")
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = pr[:, j]
    return out


def make_model(kind, params, seed, res):
    if kind == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **params)
    return RandomForestClassifier(random_state=seed, n_jobs=-1,
                                  max_samples=RF_MAX_SAMPLES[res], **RF_FIXED, **params)


def grid_search(kind, grid, Xtr, ytr, Xva, yva, seed, res):
    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        t0 = time.time()
        m = make_model(kind, params, seed, res).fit(Xtr, ytr)
        fit_s = time.time() - t0
        ev = evaluate(yva, m.predict(Xva))
        rows.append({"model": kind, **{k: str(v) for k, v in params.items()},
                     "val_macro_f1_present": ev["macro_f1_present_classes"],
                     "val_balanced_accuracy": ev["balanced_accuracy"],
                     "val_accuracy": ev["accuracy"],
                     "val_f1_congested": ev["per_class"]["congested"]["f1"],
                     "val_f1_severe": ev["per_class"]["severe"]["f1"],
                     "fit_seconds": round(fit_s, 1), "_params": params})
        print(f"   {kind} {params} -> val macro-F1={rows[-1]['val_macro_f1_present']:.4f} "
              f"bal-acc={rows[-1]['val_balanced_accuracy']:.4f} ({fit_s:.0f}s)", flush=True)
    best = max(rows, key=lambda r: (r["val_macro_f1_present"], r["val_balanced_accuracy"]))
    return best["_params"], rows


# ----------------------------------------------------------------- bieu do
def plot_confusion(results, path, title):
    names = list(results)
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 8.2))
    for j, name in enumerate(names):
        cm = np.array(results[name]["confusion_matrix"])
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        for i, (mat, sub) in enumerate(((cm, "so mau"), (norm, "ty le theo hang (recall)"))):
            ax = axes[i, j]
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for a in range(3):
                for b in range(3):
                    txt = f"{mat[a, b]:,}" if i == 0 else f"{mat[a, b]:.2f}"
                    ax.text(b, a, txt, ha="center", va="center",
                            color="white" if norm[a, b] > 0.5 else "black", fontsize=9)
            ax.set_xticks(range(3), CLASS_NAMES)
            ax.set_yticks(range(3), CLASS_NAMES)
            ax.set_xlabel("du doan")
            ax.set_ylabel("thuc te")
            ax.set_title(f"{name}\n{sub}", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(imps, path, title, top=15):
    fig, axes = plt.subplots(1, len(imps), figsize=(7 * len(imps), 6))
    axes = np.atleast_1d(axes)
    for ax, (name, s) in zip(axes, imps.items()):
        s = s.sort_values(ascending=True).tail(top)
        ax.barh(s.index, s.values, color="#0a84e0")
        ax.set_title(f"{name} (impurity-based, top {top})")
        ax.set_xlabel("muc mo hinh dua vao bien khi du doan (khong phai nhan qua)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolution", choices=list(RESOLUTIONS), default="15min")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default="", help="hau to thu muc ket qua (de chay thu nghiem khac)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = args.resolution
    seed = cfg["seed"]
    t_start = time.time()

    run_name = f"pems08_{res}{args.tag}"
    models_dir = ROOT / cfg["paths"]["models_dir"] / run_name
    reports_dir = ROOT / cfg["paths"]["reports_dir"] / run_name
    proc_dir = ROOT / cfg["paths"]["processed_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    data_path = proc_dir / f"pems08_{res}.parquet"
    if not data_path.exists():
        sys.exit(f"Thieu {data_path}. Chay truoc: python src/ml/build_pems08_dataset.py")
    df = pd.read_parquet(data_path)
    df = df.sort_values(["sensor_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    print(f"[{res}] doc {len(df):,} dong, {df.sensor_id.nunique()} sensor")
    if (df["target_level"] < 0).any() or df["target_level"].isna().any():
        raise ValueError("Co dong khong co nhan")

    # --- dac trung (chi qua khu) + bo cac dong dau chuoi thieu lag
    X_all, valid, finfo = build_features(df, res)
    rows_before = len(df)
    X_all = X_all[valid]
    meta = df.loc[valid, ["sensor_id", "timestamp", "target_timestamp", "target_speed_kmh",
                          "current_level", "target_level"]].reset_index(drop=True)
    X_all = X_all.reset_index(drop=True)
    print(f"   dac trung: {X_all.shape[1]} cot; bo {rows_before - len(X_all):,} dong dau chuoi thieu lag")

    # --- chia theo thoi gian
    tr, va, te, split_info = time_split(meta["timestamp"], cfg, res)
    Xtr, Xva, Xte = X_all[tr], X_all[va], X_all[te]
    ytr, yva, yte = (meta.loc[m, "target_level"].values for m in (tr, va, te))
    # An toan: moi timestamp thuoc dung mot tap, khong chong lan
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not np.any(meta.loc[a, "timestamp"].isin(meta.loc[b, "timestamp"]))
    assert meta.loc[tr, "target_timestamp"].max() < meta.loc[va, "timestamp"].min()
    assert meta.loc[va, "target_timestamp"].max() < meta.loc[te, "timestamp"].min()

    dist = {}
    for name, y in (("train", ytr), ("validation", yva), ("test", yte)):
        c = np.bincount(y, minlength=3)
        dist[name] = {"rows": int(len(y)), **{CLASS_NAMES[i]: int(c[i]) for i in CLASS_IDS}}
        print(f"   {name:10s}: {len(y):,} dong | " + ", ".join(f"{CLASS_NAMES[i]}={c[i]:,}" for i in CLASS_IDS))
    pd.DataFrame(dist).T.to_csv(reports_dir / "class_distribution.csv")

    # --- baseline
    majority = int(np.bincount(ytr, minlength=3).argmax())
    base_pred = {
        "baseline_majority": lambda y_n, cur: np.full(y_n, majority),
        "baseline_persistence": lambda y_n, cur: cur,
    }
    cur_va = meta.loc[va, "current_level"].values.astype(int)
    cur_te = meta.loc[te, "current_level"].values.astype(int)

    # --- chon tham so tren validation
    print("Chon tham so tren validation:")
    best, search = {}, []
    for kind, grid in (("decision_tree", DT_GRID), ("random_forest", RF_GRID)):
        best[kind], rows = grid_search(kind, grid, Xtr, ytr, Xva, yva, seed, res)
        search += rows
    pd.DataFrame([{k: v for k, v in r.items() if k != "_params"} for r in search]) \
        .to_csv(reports_dir / "validation_search.csv", index=False)
    print("Tham so da chon:", json.dumps({k: {a: str(b) for a, b in v.items()} for k, v in best.items()}))

    # --- huan luyen mo hinh cuoi (chi tren train) va luu
    models = {}
    for kind in ("decision_tree", "random_forest"):
        m = make_model(kind, best[kind], seed, res).fit(Xtr, ytr)
        models[kind] = m
        joblib.dump(m, models_dir / f"{kind}.joblib", compress=3)

    # --- danh gia validation (bao cao) va test (MOT LAN: du doan duoc tinh mot lan va dung lai)
    results = {"validation": {}, "test": {}}
    pred_cache = {}   # (split, ten mo hinh) -> du doan lop
    proba_cache = {}  # (split, ten mo hinh) -> xac suat
    for split, X, y, cur in (("validation", Xva, yva, cur_va), ("test", Xte, yte, cur_te)):
        for name, fn in base_pred.items():
            pred_cache[(split, name)] = fn(len(y), cur).astype("int8")
        for kind, m in models.items():
            pred_cache[(split, kind)] = m.predict(X).astype("int8")
            proba_cache[(split, kind)] = proba_full(m, X)
        for name in list(base_pred) + list(models):
            results[split][name] = evaluate(y, pred_cache[(split, name)])

    # --- bang metrics
    flat, per_class = [], []
    for split in results:
        for name, ev in results[split].items():
            flat.append({"split": split, "model": name, "n": ev["n"], "accuracy": ev["accuracy"],
                         "balanced_accuracy": ev["balanced_accuracy"],
                         "macro_f1_present_classes": ev["macro_f1_present_classes"],
                         "macro_f1_all_3_classes": ev["macro_f1_all_3_classes"]})
            for cls, v in ev["per_class"].items():
                per_class.append({"split": split, "model": name, "class": cls, **v})
    pd.DataFrame(flat).to_csv(reports_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(per_class).to_csv(reports_dir / "metrics_per_class.csv", index=False)

    # --- du doan test (dang dai) de luu / nap PostgreSQL
    pred_rows = []
    base_meta = meta.loc[te, ["sensor_id", "timestamp", "target_timestamp", "target_speed_kmh",
                              "target_level"]].reset_index(drop=True)
    base_meta["current_speed_kmh"] = Xte["speed_kmh"].values
    for name in base_pred:
        d = base_meta.copy()
        d["model_name"] = name
        d["predicted_level"] = pred_cache[("test", name)]
        d[["prob_free", "prob_congested", "prob_severe"]] = np.nan
        pred_rows.append(d)
    for kind in models:
        d = base_meta.copy()
        d["model_name"] = kind
        d["predicted_level"] = pred_cache[("test", kind)]
        d[["prob_free", "prob_congested", "prob_severe"]] = proba_cache[("test", kind)]
        pred_rows.append(d)
    preds = pd.concat(pred_rows, ignore_index=True).rename(
        columns={"timestamp": "feature_timestamp", "target_level": "actual_level",
                 "target_speed_kmh": "actual_speed_kmh"})
    preds.insert(0, "data_source", "PEMS08")
    preds.insert(1, "resolution", res)
    preds["split"] = "test"
    preds.to_parquet(proc_dir / f"predictions_{res}{args.tag}_test.parquet", index=False)

    # --- bieu do
    plot_confusion(results["test"], reports_dir / "confusion_matrix_test.png",
                   f"PEMS08 {res} - du doan muc un tac t+1h - tap test (California, KHONG phai Ha Noi)")
    imps = {k: pd.Series(m.feature_importances_, index=X_all.columns) for k, m in models.items()}
    for k, s in imps.items():
        s.sort_values(ascending=False).rename("importance").to_csv(reports_dir / f"feature_importance_{k}.csv")
    plot_importance(imps, reports_dir / "feature_importance.png",
                    f"PEMS08 {res} - muc mo hinh dua vao bien (khong phai quan he nhan qua)")

    # --- luu metrics + manifest chay
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source": "PEMS08 (PeMS District 8, San Bernardino, California) - NOT Hanoi",
        "resolution": res, "horizon_minutes": cfg["horizon_minutes"],
        "horizon_steps": horizon_steps(cfg, res),
        "thresholds_kmh": cfg["labels"]["thresholds_kmh"],
        "seed": seed,
        "features": list(X_all.columns), "feature_info": finfo,
        "feature_rules": "chi dung thong tin <= t cua cung sensor; khong thoi tiet, khong incident; "
                         "khong sensor_id; khong imputation/scaling (khong co buoc nao fit tren toan bo du lieu)",
        "rows": {"labeled_rows": rows_before, "after_dropping_missing_lags": int(len(X_all)),
                 "dropped_missing_lags": int(rows_before - len(X_all))},
        "split": split_info, "class_distribution": dist,
        "baselines": {"majority_class_from_train": CLASS_NAMES[majority],
                      "persistence": "du doan muc un tac tai t+1h = muc hien tai tai t"},
        "search": {"decision_tree_grid": {k: [str(x) for x in v] for k, v in DT_GRID.items()},
                   "random_forest_grid": {k: [str(x) for x in v] for k, v in RF_GRID.items()},
                   "random_forest_fixed": {**RF_FIXED, "max_samples": RF_MAX_SAMPLES[res]},
                   "selection_metric": "validation macro-F1 tren cac lop co mau (tie-break: balanced accuracy)"},
        "chosen_params": {k: {a: (None if b is None else b) for a, b in v.items()} for k, v in best.items()},
        "final_fit": "huan luyen lai tren train voi tham so da chon; validation chi dung de chon tham so; "
                     "test chi danh gia mot lan",
        "metrics": results,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                        "joblib": joblib.__version__},
        "artifacts": {
            "models": [str(p.relative_to(ROOT)).replace("\\", "/") for p in models_dir.glob("*.joblib")],
            "predictions": f"{cfg['paths']['processed_dir']}/predictions_{res}{args.tag}_test.parquet",
            "reports_dir": str(reports_dir.relative_to(ROOT)).replace("\\", "/"),
        },
        "runtime_seconds": round(time.time() - t_start, 1),
        "xgboost": "khong dung (thu vien khong co san trong moi truong du an)",
    }
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2, default=str)
    with open(models_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump({"features": list(X_all.columns), "classes": CLASS_NAMES}, f, indent=2)

    print("\n=== TEST (dung mot lan) ===")
    print(pd.DataFrame([r for r in flat if r["split"] == "test"]).round(4).to_string(index=False))
    print(f"\nXong trong {time.time() - t_start:.0f}s. Bao cao: {reports_dir}")


if __name__ == "__main__":
    main()
