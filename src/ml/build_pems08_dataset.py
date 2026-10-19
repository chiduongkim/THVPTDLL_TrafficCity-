"""
Xay bo du lieu benchmark PEMS08 cho bai toan du doan un tac tai t + 1 gio.

PEMS08 la du lieu cam bien duong cao toc PeMS District 8 (San Bernardino,
California, Hoa Ky), 170 cam bien, 5 phut/lan, 1/7/2016 - 31/8/2016.
KHONG phai du lieu Ha Noi va khong duoc ghep voi thoi tiet Ha Noi.

Dau ra (trong paths.processed_dir cua config/congestion.json):
  pems08_5min.parquet   du lieu 5 phut + nhan t+12 buoc
  pems08_15min.parquet  du lieu tong hop 15 phut + nhan t+4 buoc
  manifest_pems08.json  nguon, don vi, so dong truoc/sau xu ly, kiem tra chat luong

Chay:  python src/ml/build_pems08_dataset.py
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASS_NAMES, RESOLUTIONS, ROOT, horizon_steps,  # noqa: E402
                    load_config, speed_to_level)

# Gioi han hop ly de kiem tra truoc khi gan nhan
VALID_SPEED_MPH = (0.0, 120.0)   # toc do phai > 0 va <= 120 mph
VALID_OCCUPANCY = (0.0, 1.0)     # ty le chiem dung, 0..1
MAX_FLOW_PER_5MIN = 3000         # tong luu luong 5 phut cua 1 tram (tat ca lan)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(cfg):
    """Doc NPZ (T, N, 3) va dung bang 5 phut o dang dai: sensor_id, timestamp, flow, occupancy, speed_mph."""
    ds = cfg["dataset"]
    path = ROOT / ds["source_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Khong thay {path}. Tai tu {ds['source_url']} va dat vao data/external/pems08/")
    z = np.load(path)
    if list(z.files) != ["data"]:
        raise ValueError(f"NPZ co khoa {z.files}, du kien ['data']")
    data = z["data"]
    if data.ndim != 3 or data.shape[2] != len(ds["channels"]):
        raise ValueError(f"Hinh dang bat thuong: {data.shape}")
    return data


def quality_checks(data, cfg):
    """Kiem tra truoc khi gan nhan. Dung ngay neu co gia tri bat hop ly."""
    T, N, _ = data.shape
    flow, occ, speed = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    zero_flow = flow == 0
    q = {
        "shape_T_N_C": [int(T), int(N), 3],
        "nan_cells": int(np.isnan(data).sum()),
        "inf_cells": int(np.isinf(data).sum()),
        "negative_flow_cells": int((flow < 0).sum()),
        "flow_above_max_cells": int((flow > MAX_FLOW_PER_5MIN).sum()),
        "occupancy_out_of_range_cells": int(((occ < VALID_OCCUPANCY[0]) | (occ > VALID_OCCUPANCY[1])).sum()),
        "speed_out_of_range_cells": int(((speed <= VALID_SPEED_MPH[0]) | (speed > VALID_SPEED_MPH[1])).sum()),
        "zero_flow_cells": int(zero_flow.sum()),
        "zero_flow_cells_pct": round(float(zero_flow.mean() * 100), 4),
        "zero_flow_and_zero_occupancy_cells": int((zero_flow & (occ == 0)).sum()),
        "flow_zero_but_occupancy_positive_cells": int((zero_flow & (occ > 0)).sum()),
        "speed_when_zero_flow_median_mph": float(np.median(speed[zero_flow])) if zero_flow.any() else None,
        "sensors_with_zero_flow_share_gt_10pct": int((zero_flow.mean(0) > 0.10).sum()),
        "timesteps_with_all_sensors_zero_flow": int(zero_flow.all(1).sum()),
        "min_speed_mph": float(speed.min()), "max_speed_mph": float(speed.max()),
        "min_flow": float(flow.min()), "max_flow": float(flow.max()),
        "min_occupancy": float(occ.min()), "max_occupancy": float(occ.max()),
    }
    hard_fail = ["nan_cells", "inf_cells", "negative_flow_cells", "flow_above_max_cells",
                 "occupancy_out_of_range_cells", "speed_out_of_range_cells"]
    bad = {k: q[k] for k in hard_fail if q[k] > 0}
    if bad:
        raise ValueError(f"Du lieu co gia tri bat hop ly, dung lai: {bad}")
    return q


def to_long(data, cfg, group):
    """
    Chuyen (T, N, 3) thanh bang dai da sap xep theo sensor_id, timestamp.
    group = 1: giu nguyen 5 phut. group = 3: gop 3 buoc 5 phut thanh 15 phut:
      flow = tong, occupancy = trung binh, speed = trung binh (mph), dem so o flow=0.
    Timestamp = moc BAT DAU cua khoang [t, t + step).
    """
    ds = cfg["dataset"]
    T, N, _ = data.shape
    if T % group != 0:
        raise ValueError(f"T={T} khong chia het cho {group}")
    t5 = pd.date_range(ds["start_timestamp"], periods=T, freq=f"{ds['step_minutes']}min")
    flow, occ, speed = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    zero = (flow == 0).astype("int16")
    if group > 1:
        M = T // group
        flow = flow.reshape(M, group, N).sum(1)
        occ = occ.reshape(M, group, N).mean(1)
        speed = speed.reshape(M, group, N).mean(1)
        zero = zero.reshape(M, group, N).sum(1)
        ts = t5[::group]
    else:
        ts = t5
    M = len(ts)
    df = pd.DataFrame({
        "sensor_id": np.repeat(np.arange(N, dtype="int16"), M),
        "timestamp": np.tile(ts.values, N),
        "flow": flow.T.ravel().astype("float32"),
        "occupancy": occ.T.ravel().astype("float32"),
        "speed_mph": speed.T.ravel().astype("float32"),
        "zero_flow_steps": zero.T.ravel().astype("int8"),
    })
    df["speed_kmh"] = (df["speed_mph"].astype("float64") * ds["mph_to_kmh"]).astype("float32")
    return df


def add_labels(df, cfg, resolution):
    """
    Nhan = muc un tac cua CHINH sensor do tai t + 1 gio, lay bang cach dich chuoi
    trong tung sensor (da sap xep). Dong cuoi moi sensor khong co tuong lai bi loai,
    KHONG backfill/forward-fill target.
    """
    h = horizon_steps(cfg, resolution)
    step = RESOLUTIONS[resolution]["step_minutes"]
    df = df.sort_values(["sensor_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    if df.duplicated(["sensor_id", "timestamp"]).any():
        raise ValueError("Co timestamp trung trong cung sensor")

    g = df.groupby("sensor_id", sort=False)
    df["target_timestamp"] = g["timestamp"].shift(-h)
    df["target_speed_kmh"] = g["speed_kmh"].shift(-h)
    before = len(df)
    df = df[df["target_timestamp"].notna()].copy()
    dropped = before - len(df)

    # Kiem tra cach dich dung 1 gio, khong lech do thieu dong
    gap = (df["target_timestamp"] - df["timestamp"]).dt.total_seconds().div(60)
    if not (gap == cfg["horizon_minutes"]).all():
        raise ValueError("Co dong ma target_timestamp - timestamp khac horizon")

    df["current_level"] = speed_to_level(df["speed_kmh"].values, cfg)
    df["target_level"] = speed_to_level(df["target_speed_kmh"].values, cfg)
    df["target_speed_kmh"] = df["target_speed_kmh"].astype("float32")
    df = df.reset_index(drop=True)
    return df, {"horizon_steps": h, "step_minutes": step, "rows_before_label": before,
                "rows_dropped_no_future": dropped, "rows_after": len(df)}


def class_counts(series):
    vc = series.value_counts().reindex([0, 1, 2], fill_value=0)
    n = int(vc.sum())
    return {CLASS_NAMES[i]: {"count": int(vc[i]), "pct": round(100 * vc[i] / n, 4)} for i in range(3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = ROOT / cfg["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_raw(cfg)
    q = quality_checks(data, cfg)
    print("Kiem tra chat luong:", json.dumps({k: q[k] for k in
          ("shape_T_N_C", "nan_cells", "zero_flow_cells", "zero_flow_cells_pct")}))

    ds = cfg["dataset"]
    src = ROOT / ds["source_file"]
    dist = ROOT / ds["distance_file"]
    dist_df = pd.read_csv(dist)
    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": ds["name"],
        "geography": "PeMS District 8, San Bernardino, California, USA (highway sensors). NOT Hanoi.",
        "traffic_is_measured": True,
        "traffic_is_simulated": False,
        "source": {
            "file": ds["source_file"], "sha256": sha256(src), "bytes": src.stat().st_size,
            "download_url": ds["source_url"],
            "provenance": "File phan phoi boi ma nguon ASTGCN (Guo et al., AAAI 2019); cung kich thuoc "
                          "byte voi ban trong cac kho STFGNN, STD-MAE, STSGCN_Pytorch. Khong tai truc tiep tu PeMS.",
            "documented_by_source": "170 detectors, July to August 2016, 3 features: flow, occupy, speed (README ASTGCN)",
        },
        "period": {
            "start": ds["start_timestamp"], "step_minutes": ds["step_minutes"],
            "timesteps_5min": int(data.shape[0]),
            "end_last_step_start": str(pd.Timestamp(ds["start_timestamp"]) +
                                       pd.Timedelta(minutes=ds["step_minutes"] * (data.shape[0] - 1))),
            "timestamps_note": "File NPZ KHONG chua timestamp. Moc thoi gian duoc SUY RA tu tai lieu nguon "
                               "(1/7/2016 -> 31/8/2016, 62 ngay x 288 buoc). Kiem tra: ngay luu luong thap nhat la "
                               "4/7/2016 (le Quoc khanh My) va cac ngay Chu nhat; mau tuan khop lich. "
                               "Gio la gio dia phuong Bac My (khong luu mui gio trong file).",
        },
        "sensors": {
            "count": int(data.shape[1]),
            "id_note": "sensor_id = chi so cot 0..169 trong NPZ (giu nguyen thu tu goc). Ma tram PeMS that "
                       "khong co trong bo du lieu nay; khong gan ten duong, toa do hay camera nao.",
            "distance_file": {"file": ds["distance_file"], "sha256": sha256(dist), "rows": int(len(dist_df)),
                              "columns": list(dist_df.columns),
                              "note": "Danh sach canh from,to,cost theo chi so sensor 0..169. Don vi cua cost KHONG "
                                      "duoc nguon mo ta. Khong dung trong mo hinh."},
        },
        "channels_in_npz": {
            "order": ds["channels"],
            "evidence": "kenh 0 la so nguyen 0..1147 (flow); kenh 1 trong 0..0.90 (occupancy); "
                        "kenh 2 trong 3.0..82.3, trung vi 64.9 (speed).",
        },
        "units": {
            "flow": "so xe moi 5 phut moi tram (tong cac lan) - theo quy uoc PeMS; file khong ghi don vi",
            "occupancy": "ty le 0..1",
            "speed_in_file": "mph (SUY RA: quy uoc PeMS va trung vi 64.9 ~ gioi han toc do 65 mph; file khong ghi don vi)",
            "conversion": f"speed_kmh = speed_mph * {ds['mph_to_kmh']}",
        },
        "aggregation_15min": {
            "rule": "gop 3 buoc 5 phut lien tiep, timestamp = moc bat dau khoang",
            "flow": "tong (xe/15 phut)", "occupancy": "trung binh so hoc",
            "speed": "trung binh so hoc (mph) roi doi sang km/h; khong gia quyen theo flow",
            "zero_flow_steps": "so buoc 5 phut co flow = 0 trong khoang (0..3), cot chat luong",
        },
        "labels": {
            "classes": {"0": "free (>26 km/h)", "1": "congested (16..26 km/h, gom hai dau)", "2": "severe (<16 km/h)"},
            "thresholds_kmh": cfg["labels"]["thresholds_kmh"],
            "horizon_minutes": cfg["horizon_minutes"],
            "method": "dich chuoi toc do cua tung sensor (da sap xep) di dung 1 gio; loai dong cuoi khong co tuong lai; "
                      "khong backfill/forward-fill",
            "note": "Nguong duoc dat cho do thi (km/h). Tren cao toc PEMS08 toc do thuong 60-70 mph "
                    "(~100-110 km/h) nen hai lop un rat hiem - xem class_distribution.",
        },
        "quality_checks": q,
        "derived_columns": {
            "speed_kmh": "doi don vi tu speed_mph",
            "zero_flow_steps": "dem o flow = 0 (co the la khong co xe hoac khong do duoc; khong phan biet duoc)",
            "current_level": "muc un tac tu speed_kmh tai t",
            "target_timestamp,target_speed_kmh,target_level": "gia tri cua cung sensor tai t + 1 gio",
        },
        "excluded": {
            "weather": "PEMS08 khong kem du lieu thoi tiet California; khong ghep thoi tiet Ha Noi.",
            "incident_and_vehicle_class": "Khong co trong PEMS08, khong dung.",
            "hanoi_data": "data/raw/* (Ha Noi, giao thong mo phong) la bo rieng, khong tham gia mo hinh nay.",
        },
        "datasets": {},
    }

    for res, spec in RESOLUTIONS.items():
        long_df = to_long(data, cfg, spec["group"])
        rows_before = len(long_df)
        labeled, info = add_labels(long_df, cfg, res)
        path = out_dir / f"pems08_{res}.parquet"
        labeled.to_parquet(path, index=False)
        info.update({
            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "rows_before_processing": rows_before,
            "timestamps": int(long_df["timestamp"].nunique()),
            "first_timestamp": str(labeled["timestamp"].min()),
            "last_feature_timestamp": str(labeled["timestamp"].max()),
            "last_target_timestamp": str(labeled["target_timestamp"].max()),
            "sensors": int(labeled["sensor_id"].nunique()),
            "columns": list(labeled.columns),
            "current_level_distribution": class_counts(labeled["current_level"]),
            "target_level_distribution": class_counts(labeled["target_level"]),
            "rows_with_missing_target_level": int((labeled["target_level"] < 0).sum()),
        })
        manifest["datasets"][res] = info
        print(f"[{res}] {rows_before:,} dong -> {len(labeled):,} dong co nhan "
              f"(loai {info['rows_dropped_no_future']:,} dong cuoi chuoi) -> {path.name}")
        print(f"       target_level: " + ", ".join(
            f"{k}={v['count']:,} ({v['pct']:.3f}%)" for k, v in info["target_level_distribution"].items()))

    with open(out_dir / "manifest_pems08.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("Da ghi", out_dir / "manifest_pems08.json")


if __name__ == "__main__":
    main()
