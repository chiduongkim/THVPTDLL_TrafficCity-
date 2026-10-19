"""
Nap du lieu tu CSV vao PostgreSQL bang psycopg2.
Thay the cho Spark JDBC tren Windows.
"""
import csv
import psycopg2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_RAW = os.path.join(PROJECT_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_DIR, "data", "processed")

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "traffic_db",
    "user": "admin",
    "password": "admin123"
}


def connect():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    return conn


def load_cameras(conn):
    print("\n[1/5] Nap cameras...")
    cur = conn.cursor()
    # Xoa du lieu cu (giu schema)
    cur.execute("DELETE FROM traffic_summary")
    cur.execute("DELETE FROM violations")
    cur.execute("DELETE FROM traffic_data")
    cur.execute("DELETE FROM weather_data")
    # Khong xoa cameras vi schema.sql da insert san
    
    filepath = os.path.join(DATA_RAW, "cameras.csv")
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute("""
                INSERT INTO cameras (camera_id, camera_name, location, latitude, longitude, road_name, number_of_lanes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (camera_id) DO UPDATE SET
                    camera_name = EXCLUDED.camera_name,
                    location = EXCLUDED.location
            """, (
                int(row["camera_id"]),
                row["camera_name"],
                row["location"],
                float(row["latitude"]),
                float(row["longitude"]),
                row["road_name"],
                int(row["number_of_lanes"])
            ))
    cur.execute("SELECT count(*) FROM cameras")
    print(f"  -> {cur.fetchone()[0]} cameras")


def load_weather(conn):
    print("\n[2/5] Nap weather_data...")
    cur = conn.cursor()
    filepath = os.path.join(DATA_RAW, "weather_data.csv")
    count = 0
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute("""
                INSERT INTO weather_data (recorded_at, temperature, humidity, rainfall, weather_type, wind_speed, weather_code, location)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                row["timestamp"],
                float(row["temperature"]),
                float(row["humidity"]),
                float(row["rainfall"]),
                row["weather_type"],
                float(row["wind_speed"]),
                int(row["weather_code"]),
                "Ha Noi"
            ))
            count += 1
    print(f"  -> {count} ban ghi thoi tiet")


def load_violations(conn):
    print("\n[3/5] Nap violations...")
    cur = conn.cursor()
    filepath = os.path.join(DATA_RAW, "violations.csv")
    count = 0
    batch = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.append((
                int(row["camera_id"]),
                row["timestamp"],
                row["vehicle_type"],
                row["violation_type"],
                float(row["speed"])
            ))
            count += 1
            if len(batch) >= 1000:
                cur.executemany("""
                    INSERT INTO violations (camera_id, recorded_at, vehicle_type, violation_type, speed)
                    VALUES (%s, %s, %s, %s, %s)
                """, batch)
                batch = []
                if count % 10000 == 0:
                    print(f"    {count:,} vi pham...")
    if batch:
        cur.executemany("""
            INSERT INTO violations (camera_id, recorded_at, vehicle_type, violation_type, speed)
            VALUES (%s, %s, %s, %s, %s)
        """, batch)
    print(f"  -> {count:,} vi pham")


def load_summary(conn):
    print("\n[4/5] Nap traffic_summary...")
    cur = conn.cursor()
    filepath = os.path.join(DATA_PROCESSED, "traffic_summary.csv")
    count = 0
    batch = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        print(f"  Cot: {cols}")
        for row in reader:
            cam_id = int(row.get("camera_id", 0))
            summary_date = row.get("summary_date", row.get("date", "2024-09-01"))
            hour_of_day = int(row.get("hour_of_day", 0))
            total_vehicles = int(row.get("total_vehicles", 0))
            car_count = int(row.get("car_count", 0))
            motorcycle_count = int(row.get("motorcycle_count", 0))
            bicycle_count = int(row.get("bicycle_count", 0))
            truck_count = int(row.get("truck_count", 0))
            bus_count = int(row.get("bus_count", 0))
            avg_speed = float(row.get("avg_speed", 0))
            violation_count = int(row.get("violation_count", 0))
            congestion = row.get("congestion_level", "low")
            weather = row.get("weather_type", "clear")
            temp = float(row.get("temperature", 0)) if row.get("temperature") else None
            rain = float(row.get("rainfall", 0)) if row.get("rainfall") else None

            batch.append((
                cam_id, summary_date, hour_of_day, total_vehicles,
                car_count, motorcycle_count, bicycle_count, truck_count, bus_count,
                avg_speed, violation_count, congestion, weather, temp, rain
            ))
            count += 1
            if len(batch) >= 500:
                cur.executemany("""
                    INSERT INTO traffic_summary 
                    (camera_id, summary_date, hour_of_day, total_vehicles,
                     car_count, motorcycle_count, bicycle_count, truck_count, bus_count,
                     avg_speed, violation_count, congestion_level, weather_type, temperature, rainfall)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, batch)
                batch = []
    if batch:
        cur.executemany("""
            INSERT INTO traffic_summary 
            (camera_id, summary_date, hour_of_day, total_vehicles,
             car_count, motorcycle_count, bicycle_count, truck_count, bus_count,
             avg_speed, violation_count, congestion_level, weather_type, temperature, rainfall)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, batch)
    print(f"  -> {count:,} ban ghi summary")


def verify(conn):
    print("\n[5/5] Kiem tra ket qua...")
    cur = conn.cursor()
    tables = ["cameras", "weather_data", "violations", "traffic_summary"]
    for t in tables:
        cur.execute(f"SELECT count(*) FROM {t}")
        print(f"  {t}: {cur.fetchone()[0]:,} ban ghi")
    
    # Thong ke nhanh
    cur.execute("SELECT congestion_level, count(*) FROM traffic_summary GROUP BY congestion_level ORDER BY count(*) DESC")
    print("\n  Muc do un tac:")
    for row in cur.fetchall():
        print(f"    {row[0]}: {row[1]:,}")
    
    cur.execute("SELECT weather_type, ROUND((SUM(avg_speed * total_vehicles) / NULLIF(SUM(total_vehicles), 0))::numeric, 1), SUM(violation_count) FROM traffic_summary GROUP BY weather_type ORDER BY 2")
    print("\n  Thoi tiet vs Toc do:")
    for row in cur.fetchall():
        print(f"    {row[0]}: {row[1]} km/h, {row[2]:,} vi pham")


def main():
    print("=" * 55)
    print("  NAP DU LIEU VAO POSTGRESQL - GIAO THONG HA NOI")
    print("=" * 55)
    
    conn = connect()
    print(f"Da ket noi PostgreSQL (port {DB_CONFIG['port']})")
    
    try:
        load_cameras(conn)
        load_weather(conn)
        load_violations(conn)
        load_summary(conn)
        verify(conn)
        print("\n" + "=" * 55)
        print("  HOAN TAT! Du lieu da san sang cho Power BI!")
        print("=" * 55)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
