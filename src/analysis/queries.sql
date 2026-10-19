-- =============================================
-- CAC CAU TRUY VAN PHAN TICH GIAO THONG HA NOI
-- =============================================
-- Ghi chu: toc do trung binh luon tinh CO TRONG SO theo so xe
--   SUM(avg_speed * total_vehicles) / SUM(total_vehicles)
-- vi traffic_summary la tong hop theo (camera, ngay, gio); AVG(avg_speed) thuong
-- se coi 1 gio dem vang xe (vai chuc xe) ngang 1 gio cao diem (hang nghin xe).

-- ============ CAU 1: Khung gio dong xe nhat ============
SELECT
    hour_of_day,
    SUM(total_vehicles) AS total,
    ROUND((SUM(avg_speed * total_vehicles) / NULLIF(SUM(total_vehicles), 0))::numeric, 1) AS avg_speed,
    ROUND(AVG((congestion_level IN ('high', 'critical'))::int)::numeric * 100, 1) AS congestion_pct
FROM traffic_summary
GROUP BY hour_of_day
ORDER BY total DESC;

-- ============ CAU 2: Khu vuc un tac nhat ============
SELECT
    c.camera_name,
    c.road_name,
    c.location,
    ROUND((SUM(ts.avg_speed * ts.total_vehicles) / NULLIF(SUM(ts.total_vehicles), 0))::numeric, 1) AS avg_speed,
    SUM(ts.total_vehicles) AS total_vehicles,
    COUNT(CASE WHEN ts.congestion_level IN ('high', 'critical')
          THEN 1 END) AS congested_hours,
    SUM(ts.violation_count) AS total_violations
FROM traffic_summary ts
JOIN cameras c ON ts.camera_id = c.camera_id
GROUP BY c.camera_name, c.road_name, c.location
ORDER BY congested_hours DESC;

-- ============ CAU 3: Ty le loai phuong tien ============
SELECT
    'Oto' AS vehicle_type, SUM(car_count) AS total
FROM traffic_summary
UNION ALL
SELECT 'Xe may', SUM(motorcycle_count) FROM traffic_summary
UNION ALL
SELECT 'Xe dap', SUM(bicycle_count) FROM traffic_summary
UNION ALL
SELECT 'Xe tai', SUM(truck_count) FROM traffic_summary
UNION ALL
SELECT 'Xe bus', SUM(bus_count) FROM traffic_summary
ORDER BY total DESC;

-- ============ CAU 4: Toc do theo gio ============
SELECT
    hour_of_day,
    ROUND((SUM(avg_speed * total_vehicles) / NULLIF(SUM(total_vehicles), 0))::numeric, 1) AS avg_speed
FROM traffic_summary
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- ============ CAU 5: Vi pham nhieu nhat ============
SELECT
    violation_type,
    COUNT(*) AS total,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM violations
WHERE violation_type IS NOT NULL AND violation_type != ''
GROUP BY violation_type
ORDER BY total DESC;

-- ============ CAU 6: Anh huong thoi tiet ============
SELECT
    weather_type,
    ROUND((SUM(avg_speed * total_vehicles) / NULLIF(SUM(total_vehicles), 0))::numeric, 1) AS avg_speed,
    SUM(total_vehicles) AS total_vehicles,
    SUM(violation_count) AS total_violations,
    ROUND(SUM(violation_count) * 100.0 / NULLIF(SUM(total_vehicles), 0), 1) AS violation_rate_pct,
    ROUND(AVG(total_vehicles)::numeric, 0) AS avg_vehicles_per_hour
FROM traffic_summary
WHERE weather_type IS NOT NULL
GROUP BY weather_type
ORDER BY avg_speed;

-- ============ CAU 7: So sanh ngay thuong vs cuoi tuan ============
SELECT
    CASE
        WHEN EXTRACT(DOW FROM summary_date::date) IN (0, 6) THEN 'Cuoi tuan'
        ELSE 'Ngay thuong'
    END AS day_type,
    ROUND(AVG(total_vehicles)::numeric, 0) AS avg_vehicles,
    ROUND((SUM(avg_speed * total_vehicles) / NULLIF(SUM(total_vehicles), 0))::numeric, 1) AS avg_speed,
    SUM(violation_count) AS total_violations
FROM traffic_summary
GROUP BY day_type;

-- ============ CAU 8: Xu huong theo ngay ============
SELECT
    summary_date,
    SUM(total_vehicles) AS total_vehicles,
    ROUND((SUM(avg_speed * total_vehicles) / NULLIF(SUM(total_vehicles), 0))::numeric, 1) AS avg_speed,
    SUM(violation_count) AS violations
FROM traffic_summary
GROUP BY summary_date
ORDER BY summary_date;
