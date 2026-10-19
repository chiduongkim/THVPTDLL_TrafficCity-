-- =============================================
-- SCHEMA: He thong phan tich giao thong Ha Noi
-- =============================================

-- Bang cameras: thong tin vi tri camera
CREATE TABLE IF NOT EXISTS cameras (
    camera_id SERIAL PRIMARY KEY,
    camera_name VARCHAR(50) NOT NULL,
    location VARCHAR(200),
    latitude FLOAT,
    longitude FLOAT,
    road_name VARCHAR(200),
    number_of_lanes INT DEFAULT 2
);

-- Bang traffic_data: du lieu giao thong chi tiet
CREATE TABLE IF NOT EXISTS traffic_data (
    id BIGSERIAL PRIMARY KEY,
    camera_id INT REFERENCES cameras(camera_id),
    recorded_at TIMESTAMP NOT NULL,
    vehicle_type VARCHAR(20),
    speed FLOAT,
    direction VARCHAR(10),
    vehicle_count INT DEFAULT 1,
    traffic_density FLOAT
);

-- Bang violations: vi pham giao thong
CREATE TABLE IF NOT EXISTS violations (
    id BIGSERIAL PRIMARY KEY,
    camera_id INT REFERENCES cameras(camera_id),
    recorded_at TIMESTAMP NOT NULL,
    vehicle_type VARCHAR(20),
    violation_type VARCHAR(50),
    speed FLOAT,
    evidence_path VARCHAR(500)
);

-- Bang weather_data: du lieu thoi tiet
CREATE TABLE IF NOT EXISTS weather_data (
    id BIGSERIAL PRIMARY KEY,
    recorded_at TIMESTAMP NOT NULL,
    temperature FLOAT,
    humidity FLOAT,
    rainfall FLOAT,
    weather_type VARCHAR(20),
    wind_speed FLOAT,
    weather_code INT,
    location VARCHAR(200)
);

-- Bang traffic_summary: tong hop theo gio (Spark ghi vao)
CREATE TABLE IF NOT EXISTS traffic_summary (
    id BIGSERIAL PRIMARY KEY,
    camera_id INT REFERENCES cameras(camera_id),
    summary_date DATE NOT NULL,
    hour_of_day INT NOT NULL,
    total_vehicles INT,
    car_count INT DEFAULT 0,
    motorcycle_count INT DEFAULT 0,
    bicycle_count INT DEFAULT 0,
    truck_count INT DEFAULT 0,
    bus_count INT DEFAULT 0,
    avg_speed FLOAT,
    violation_count INT DEFAULT 0,
    congestion_level VARCHAR(20),
    weather_type VARCHAR(20),
    temperature FLOAT,
    rainfall FLOAT
);

-- Index de query nhanh
CREATE INDEX IF NOT EXISTS idx_traffic_time ON traffic_data(recorded_at);
CREATE INDEX IF NOT EXISTS idx_traffic_camera ON traffic_data(camera_id);
CREATE INDEX IF NOT EXISTS idx_violations_time ON violations(recorded_at);
CREATE INDEX IF NOT EXISTS idx_violations_camera ON violations(camera_id);
CREATE INDEX IF NOT EXISTS idx_weather_time ON weather_data(recorded_at);
CREATE INDEX IF NOT EXISTS idx_summary_date ON traffic_summary(summary_date, hour_of_day);
CREATE INDEX IF NOT EXISTS idx_summary_camera ON traffic_summary(camera_id);

-- Du lieu mau cameras tai Ha Noi
INSERT INTO cameras (camera_name, location, latitude, longitude, road_name, number_of_lanes)
VALUES
    ('CAM_HN01', 'Nga tu Dai Co Viet - Giai Phong', 21.0024, 105.8432, 'Dai Co Viet', 4),
    ('CAM_HN02', 'Nga tu Nguyen Trai - Khuat Duy Tien', 20.9963, 105.8005, 'Nguyen Trai', 6),
    ('CAM_HN03', 'Cau Vinh Tuy', 20.9978, 105.8752, 'Cau Vinh Tuy', 4),
    ('CAM_HN04', 'Nga tu Cau Giay', 21.0285, 105.7960, 'Cau Giay', 4),
    ('CAM_HN05', 'Nga tu Tran Duy Hung - Hoang Minh Giam', 21.0105, 105.8015, 'Tran Duy Hung', 6)
ON CONFLICT DO NOTHING;
