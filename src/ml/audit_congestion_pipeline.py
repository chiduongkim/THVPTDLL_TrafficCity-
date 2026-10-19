"""
Kiem tra doc lap pipeline du doan un tac (khong tin vao code xay dung).

  A. Nhan: tinh lai muc un tac t+1h truc tiep tu file NPZ goc va so voi cot target_level.
  B. Dong cuoi chuoi: moi sensor mat dung `horizon` dong cuoi, khong co nhan backfill.
  C. Dac trung nhan qua: doi toan bo gia tri SAU mot moc thoi gian bat ky roi tinh lai
     dac trung - cac dong co timestamp <= moc phai giu nguyen.
  D. Chia tap: khong chong timestamp, nhan cua tap truoc nam truoc tap sau.
  E. Du doan test: khong trung khoa, so dong dung.

Chay:  python src/ml/audit_congestion_pipeline.py [--resolution 15min]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESOLUTIONS, ROOT, horizon_steps, load_config, speed_to_level  # noqa: E402
from train_congestion_models import build_features  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" - {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def audit(res, cfg):
    spec = RESOLUTIONS[res]
    ds = cfg["dataset"]
    h = horizon_steps(cfg, res)
    proc = ROOT / cfg["paths"]["processed_dir"]
    df = pd.read_parquet(proc / f"pems08_{res}.parquet")
    print(f"\n===== {res}: {len(df):,} dong =====")

    # A. Nhan tinh lai tu NPZ goc
    raw = np.load(ROOT / ds["source_file"])["data"]
    T, N, _ = raw.shape
    g = spec["group"]
    speed = raw[:, :, 2].reshape(T // g, g, N).mean(1) * ds["mph_to_kmh"]     # (M, N) km/h
    M = speed.shape[0]
    levels = speed_to_level(speed, cfg)                                       # (M, N)
    start = pd.Timestamp(ds["start_timestamp"])
    step = pd.Timedelta(minutes=spec["step_minutes"])
    m_idx = ((df["timestamp"] - start) / step).round().astype(int).values
    s_idx = df["sensor_id"].values.astype(int)
    expected = levels[m_idx + h, s_idx]
    check("A1 target_level == nhan tinh lai tu NPZ goc (toan bo dong)",
          np.array_equal(expected, df["target_level"].values), f"{len(df):,} dong")
    check("A2 current_level == nhan tinh lai tai t",
          np.array_equal(levels[m_idx, s_idx], df["current_level"].values))
    check("A3 target_speed_kmh khop NPZ goc",
          np.allclose(speed[m_idx + h, s_idx], df["target_speed_kmh"].values, atol=1e-3))
    tgt = (df["target_timestamp"] - df["timestamp"]).dt.total_seconds() / 60
    check("A4 target_timestamp - timestamp = 60 phut", bool((tgt == cfg["horizon_minutes"]).all()))

    # B. Dong cuoi chuoi
    per_sensor = df.groupby("sensor_id")["timestamp"].agg(["count", "max"])
    check("B1 moi sensor con dung M - horizon dong", bool((per_sensor["count"] == M - h).all()),
          f"{M - h} dong/sensor")
    last_ok = start + step * (M - 1 - h)
    check("B2 timestamp lon nhat moi sensor = cuoi - 1 gio", bool((per_sensor["max"] == last_ok).all()),
          str(last_ok))
    check("B3 khong co NaN trong cot nhan/target", not df[["target_timestamp", "target_speed_kmh", "target_level"]].isna().any().any())
    check("B4 khong trung (sensor_id, timestamp)", not df.duplicated(["sensor_id", "timestamp"]).any())

    # C. Dac trung nhan qua: thay doi tuong lai khong duoc anh huong dong qua khu
    sub = df[df["sensor_id"] < 6].sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    X0, v0, _ = build_features(sub, res)
    cut = sub["timestamp"].quantile(0.6, interpolation="nearest")
    pert = sub.copy()
    pert["zero_flow_steps"] = pert["zero_flow_steps"].astype("float32")
    rng = np.random.default_rng(0)
    fut = pert["timestamp"] > cut
    for c, lo, hi in (("speed_kmh", 1, 150), ("flow", 0, 900), ("occupancy", 0, 1), ("zero_flow_steps", 0, 3)):
        pert.loc[fut, c] = rng.uniform(lo, hi, fut.sum()).astype("float32")
    X1, v1, _ = build_features(pert, res)
    past = (sub["timestamp"] <= cut).values & v0 & v1
    same = np.allclose(X0.loc[past].values, X1.loc[past].values, equal_nan=True)
    check("C1 doi du lieu SAU moc thoi gian khong lam doi dac trung <= moc", same,
          f"{int(past.sum()):,} dong kiem tra, moc {cut}")
    changed = not np.allclose(X0.loc[~past & v0 & v1].values, X1.loc[~past & v0 & v1].values)
    check("C2 (doi chung) dac trung SAU moc co thay doi -> phep thu co hieu luc", changed)
    forbidden = [c for c in X0.columns if "target" in c or "future" in c or "next" in c or c in ("sensor_id",)]
    check("C3 khong co cot target/future/next/sensor_id trong dac trung", not forbidden, str(forbidden))

    # D. Chia tap tu manifest huan luyen
    mpath = ROOT / cfg["paths"]["reports_dir"] / f"pems08_{res}" / "metrics.json"
    if mpath.exists():
        mm = json.load(open(mpath, encoding="utf-8"))
        sp = mm["split"]
        tr_end, va_start = pd.Timestamp(sp["train_range"][1]), pd.Timestamp(sp["validation_range"][0])
        va_end, te_start = pd.Timestamp(sp["validation_range"][1]), pd.Timestamp(sp["test_range"][0])
        hz = pd.Timedelta(minutes=cfg["horizon_minutes"])
        check("D1 nhan train (t+1h) nam truoc validation", tr_end + hz < va_start)
        check("D2 nhan validation (t+1h) nam truoc test", va_end + hz < te_start)
        check("D3 thu tu tap: train < validation < test", tr_end < va_start <= va_end < te_start)
        check("D4 seed va tham so duoc ghi", "seed" in mm and "chosen_params" in mm)
        # E. Du doan test
        pp = proc / f"predictions_{res}_test.parquet"
        pr = pd.read_parquet(pp)
        key = ["data_source", "resolution", "model_name", "sensor_id", "feature_timestamp"]
        check("E1 du doan test khong trung khoa", not pr.duplicated(key).any(), f"{len(pr):,} dong")
        n_models = pr["model_name"].nunique()
        check("E2 moi mo hinh co cung so dong", pr.groupby("model_name").size().nunique() == 1,
              f"{n_models} mo hinh")
        check("E3 feature_timestamp cua du doan >= dau tap test",
              bool((pr["feature_timestamp"] >= te_start).all()))
        ref = pr[pr["model_name"] == "baseline_persistence"]
        look = df.set_index(["sensor_id", "timestamp"])["current_level"]
        idx = pd.MultiIndex.from_arrays([ref["sensor_id"], ref["feature_timestamp"]])
        check("E4 baseline_persistence == current_level tai t",
              np.array_equal(look.reindex(idx).values, ref["predicted_level"].values))
        # E5: vi sao ben persistence co so luong du doan bang so luong thuc te
        cur = ref["predicted_level"].value_counts().sort_index().to_dict()
        act = ref["actual_level"].value_counts().sort_index().to_dict()
        print(f"     (thong tin) persistence du doan theo lop {cur} vs thuc te {act}: "
              "hai tap chi lech nhau o hai dau khoang 1 gio cua tap test")
    else:
        print("   (bo qua D/E: chua co metrics.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", choices=list(RESOLUTIONS), default=None)
    args = ap.parse_args()
    cfg = load_config()
    for res in ([args.resolution] if args.resolution else list(RESOLUTIONS)):
        audit(res, cfg)
    print("\nKET QUA:", "TAT CA DAT" if not FAILS else f"CO LOI: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
