"""
Ham dung chung cho pipeline du doan un tac (PEMS08 benchmark).
Doc cau hinh tu config/congestion.json va gan nhan muc un tac tu toc do (km/h).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "congestion.json"

# Ma lop: 0 = thong thoang, 1 = un u, 2 = un tac nghiem trong
CLASS_NAMES = ["free", "congested", "severe"]
CLASS_IDS = [0, 1, 2]

# Moi do phan giai: so phut moi buoc, so buoc gop tu du lieu 5 phut
RESOLUTIONS = {
    "5min": {"step_minutes": 5, "group": 1},
    "15min": {"step_minutes": 15, "group": 3},
}


def load_config(path=None):
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def horizon_steps(cfg, resolution):
    """So buoc can dich de lay quan sat sau dung 1 gio (t+12 cho 5 phut, t+4 cho 15 phut)."""
    step = RESOLUTIONS[resolution]["step_minutes"]
    h = cfg["horizon_minutes"]
    if h % step != 0:
        raise ValueError(f"horizon {h} phut khong chia het cho buoc {step} phut")
    return h // step


def speed_to_level(speed_kmh, cfg):
    """
    Phan lop theo nguong trong cau hinh (don vi km/h):
      > free_above                         -> 0 free
      congested_min <= v <= congested_max  -> 1 congested
      < congested_min                      -> 2 severe
    NaN giu nguyen la -1 (khong phan lop).
    """
    t = cfg["labels"]["thresholds_kmh"]
    v = np.asarray(speed_kmh, dtype="float64")
    level = np.full(v.shape, -1, dtype="int8")
    ok = ~np.isnan(v)
    level[ok & (v > t["free_above"])] = 0
    level[ok & (v >= t["congested_min"]) & (v <= t["congested_max"])] = 1
    level[ok & (v < t["congested_min"])] = 2
    return level
