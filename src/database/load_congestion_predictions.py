"""
Nap ket qua du doan un tac (PEMS08) vao PostgreSQL.

- Ap dung migration src/database/migrations/002_congestion_predictions.sql
  (chi tao bang ml_model_runs, ml_congestion_predictions; khong dong vao bang hien co).
- Nap du doan tap test bang COPY vao bang tam roi UPSERT theo khoa duy nhat
  (data_source, resolution, model_name, sensor_id, feature_timestamp) -> chay lai an toan.
- Toan bo trong MOT giao dich: loi giua chung se rollback.

Ket noi: bien moi truong PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD
(mac dinh trung voi docker-compose: localhost:5433, traffic_db, admin).

Chay:  python src/database/load_congestion_predictions.py [--resolution 15min]
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import Json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "ml"))
from common import RESOLUTIONS, load_config  # noqa: E402

MIGRATION = Path(__file__).resolve().parent / "migrations" / "002_congestion_predictions.sql"
EXISTING_TABLES = ["cameras", "traffic_data", "violations", "weather_data", "traffic_summary"]
COLS = ["data_source", "resolution", "model_name", "sensor_id", "feature_timestamp", "target_timestamp",
        "split", "current_speed_kmh", "actual_speed_kmh", "actual_level", "predicted_level",
        "prob_free", "prob_congested", "prob_severe"]
CHUNK = 200_000


def connect():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"), port=int(os.getenv("PGPORT", "5433")),
        dbname=os.getenv("PGDATABASE", "traffic_db"), user=os.getenv("PGUSER", "admin"),
        password=os.getenv("PGPASSWORD", "admin123"))


def count_existing(cur):
    out = {}
    for t in EXISTING_TABLES:
        cur.execute("SELECT to_regclass(%s)", (t,))
        out[t] = None
        if cur.fetchone()[0]:
            cur.execute(f"SELECT count(*) FROM {t}")
            out[t] = cur.fetchone()[0]
    return out


def upsert_runs(cur, cfg, res, manifest):
    """Mot dong ml_model_runs cho moi mo hinh (ke ca baseline)."""
    test = manifest["metrics"]["test"]
    for model, metrics in test.items():
        params = manifest["chosen_params"].get(model)
        cur.execute("""
            INSERT INTO ml_model_runs (data_source, resolution, model_name, horizon_minutes,
                thresholds_kmh, params, metrics_test, split_info, run_created_utc, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (data_source, resolution, model_name) DO UPDATE SET
                horizon_minutes = EXCLUDED.horizon_minutes, thresholds_kmh = EXCLUDED.thresholds_kmh,
                params = EXCLUDED.params, metrics_test = EXCLUDED.metrics_test,
                split_info = EXCLUDED.split_info, run_created_utc = EXCLUDED.run_created_utc,
                notes = EXCLUDED.notes, loaded_at = now()
        """, ("PEMS08", res, model, cfg["horizon_minutes"], Json(cfg["labels"]["thresholds_kmh"]),
              Json(params) if params else None, Json(metrics), Json(manifest["split"]),
              manifest["created_utc"],
              "PEMS08 = cam bien cao toc California (khong phai Ha Noi); seed=%s" % manifest["seed"]))


def load_predictions(cur, df):
    """COPY tung phan vao bang tam roi UPSERT."""
    cur.execute("DROP TABLE IF EXISTS _stg_ml_pred")
    cur.execute("""
        CREATE TEMP TABLE _stg_ml_pred (LIKE ml_congestion_predictions INCLUDING DEFAULTS)
        ON COMMIT DROP
    """)
    cur.execute("ALTER TABLE _stg_ml_pred DROP COLUMN id, DROP COLUMN loaded_at")
    for start in range(0, len(df), CHUNK):
        buf = io.StringIO()
        df.iloc[start:start + CHUNK][COLS].to_csv(buf, index=False, header=False, na_rep="")
        buf.seek(0)
        cur.copy_expert(f"COPY _stg_ml_pred ({', '.join(COLS)}) FROM STDIN WITH (FORMAT csv)", buf)
    cur.execute("SELECT count(*) FROM _stg_ml_pred")
    staged = cur.fetchone()[0]
    upd = ", ".join(f"{c} = EXCLUDED.{c}" for c in COLS[5:])
    cur.execute(f"""
        INSERT INTO ml_congestion_predictions ({', '.join(COLS)})
        SELECT {', '.join(COLS)} FROM _stg_ml_pred
        ON CONFLICT (data_source, resolution, model_name, sensor_id, feature_timestamp)
        DO UPDATE SET {upd}, loaded_at = now()
    """)
    return staged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", choices=list(RESOLUTIONS), default=None)
    args = ap.parse_args()
    cfg = load_config()
    resolutions = [args.resolution] if args.resolution else list(RESOLUTIONS)

    conn = connect()
    try:
        with conn:                      # mot giao dich: commit neu thanh cong, rollback neu loi
            with conn.cursor() as cur:
                before = count_existing(cur)
                cur.execute(MIGRATION.read_text(encoding="utf-8"))
                print("Da ap dung migration 002 (IF NOT EXISTS).")
                for res in resolutions:
                    mpath = ROOT / cfg["paths"]["reports_dir"] / f"pems08_{res}" / "metrics.json"
                    ppath = ROOT / cfg["paths"]["processed_dir"] / f"predictions_{res}_test.parquet"
                    if not (mpath.exists() and ppath.exists()):
                        print(f"[{res}] bo qua: thieu {mpath.name} hoac {ppath.name}")
                        continue
                    manifest = json.loads(mpath.read_text(encoding="utf-8"))
                    df = pd.read_parquet(ppath)
                    df["feature_timestamp"] = df["feature_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
                    df["target_timestamp"] = df["target_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
                    upsert_runs(cur, cfg, res, manifest)
                    staged = load_predictions(cur, df)
                    print(f"[{res}] nap {staged:,} dong du doan ({df['model_name'].nunique()} mo hinh)")
                after = count_existing(cur)
                if before != after:
                    raise RuntimeError(f"Bang hien co bi thay doi so dong! truoc={before} sau={after}")
                cur.execute("""SELECT data_source, resolution, model_name, count(*)
                               FROM ml_congestion_predictions GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""")
                print("\nml_congestion_predictions:")
                for r in cur.fetchall():
                    print("  %s | %s | %-22s | %s dong" % (r[0], r[1], r[2], f"{r[3]:,}"))
                print("\nBang hien co (khong doi):", after)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
