-- =============================================================
-- MIGRATION 002: Du doan muc un tac (benchmark PEMS08)
-- Chi TAO MOI bang ml_*; khong DROP/ALTER/DELETE bang nao cua du an hien tai.
-- Chay lai nhieu lan an toan (IF NOT EXISTS).
--
-- LUU Y NGUON DU LIEU: sensor trong cac bang nay la tram cam bien duong cao toc
-- PeMS District 8 (San Bernardino, California, Hoa Ky). KHONG lien ket voi bang
-- cameras cua Ha Noi, khong co FK toi cameras.
-- =============================================================

CREATE TABLE IF NOT EXISTS ml_model_runs (
    id                BIGSERIAL PRIMARY KEY,
    data_source       TEXT      NOT NULL,           -- 'PEMS08'
    resolution        TEXT      NOT NULL CHECK (resolution IN ('5min', '15min')),
    model_name        TEXT      NOT NULL,           -- decision_tree | random_forest | baseline_*
    horizon_minutes   INT       NOT NULL,
    thresholds_kmh    JSONB,
    params            JSONB,
    metrics_test      JSONB,
    split_info        JSONB,
    run_created_utc   TEXT,
    notes             TEXT,
    loaded_at         TIMESTAMP NOT NULL DEFAULT now(),
    CONSTRAINT uq_ml_model_runs UNIQUE (data_source, resolution, model_name)
);

CREATE TABLE IF NOT EXISTS ml_congestion_predictions (
    id                 BIGSERIAL PRIMARY KEY,
    data_source        TEXT      NOT NULL,
    resolution         TEXT      NOT NULL CHECK (resolution IN ('5min', '15min')),
    model_name         TEXT      NOT NULL,
    sensor_id          INT       NOT NULL,          -- chi so cot 0..169 trong PEMS08, khong phai camera
    feature_timestamp  TIMESTAMP NOT NULL,          -- thoi diem t (gio dia phuong Bac My)
    target_timestamp   TIMESTAMP NOT NULL,          -- t + 1 gio
    split              TEXT      NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    current_speed_kmh  REAL,
    actual_speed_kmh   REAL,
    actual_level       SMALLINT  NOT NULL CHECK (actual_level BETWEEN 0 AND 2),
    predicted_level    SMALLINT  NOT NULL CHECK (predicted_level BETWEEN 0 AND 2),
    prob_free          REAL,
    prob_congested     REAL,
    prob_severe        REAL,
    loaded_at          TIMESTAMP NOT NULL DEFAULT now(),
    CONSTRAINT uq_ml_congestion_prediction
        UNIQUE (data_source, resolution, model_name, sensor_id, feature_timestamp),
    CONSTRAINT fk_ml_prediction_run
        FOREIGN KEY (data_source, resolution, model_name)
        REFERENCES ml_model_runs (data_source, resolution, model_name)
);

CREATE INDEX IF NOT EXISTS idx_ml_pred_time
    ON ml_congestion_predictions (data_source, resolution, model_name, feature_timestamp);
CREATE INDEX IF NOT EXISTS idx_ml_pred_sensor
    ON ml_congestion_predictions (data_source, sensor_id);

COMMENT ON TABLE ml_congestion_predictions IS
    'Du doan muc un tac t+1h tren tap TEST. Du lieu PEMS08 (California), khong phai Ha Noi.';
COMMENT ON COLUMN ml_congestion_predictions.actual_level IS
    '0=free (>26 km/h), 1=congested (16..26 km/h), 2=severe (<16 km/h); toc do goc mph doi sang km/h';
COMMENT ON TABLE ml_model_runs IS
    'Tham so, nguong va metrics test cua tung mo hinh; luu ban moi nhat cho moi (nguon, do phan giai, mo hinh).';
