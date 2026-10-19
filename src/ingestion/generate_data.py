"""
Script tao du lieu giao thong gia lap cho 5 camera tai Ha Noi,
khop CHINH XAC voi du lieu thoi tiet thuc te tu Open-Meteo CSV (Thang 9/2024).

Dau ra:
- data/raw/traffic_data.csv: Tat ca ban ghi phuong tien (~200.000 - 400.000 dong)
- data/raw/weather_data.csv: Ban ghi thoi tiet theo gio (720 dong)
- data/raw/cameras.csv: Thong tin 5 camera giao thong Ha Noi
- data/raw/violations.csv: Chi cac ban ghi vi pham giao thong
"""

import os
import sys
import csv
import random
import argparse
from datetime import datetime, timedelta

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN / PATH CONFIGURATION
# ==============================================================================
BASE_DIR = os.getenv(
    "PROJECT_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# File Open-Meteo goc (tai tu open-meteo.com, xuat CSV hourly, 1-30/9/2024, Ha Noi 21.05N 105.81E).
# Tu dong tim trong data/external/ roi toi thu muc goc du an; co the doi bang
# bien moi truong WEATHER_CSV_PATH hoac tham so --weather-file.
_WEATHER_NAME = "open-meteo-21.05N105.81E24m.csv"
_WEATHER_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "external", _WEATHER_NAME),
    os.path.join(BASE_DIR, _WEATHER_NAME),
]
DEFAULT_WEATHER_PATH = next((p for p in _WEATHER_CANDIDATES if os.path.exists(p)), _WEATHER_CANDIDATES[0])
WEATHER_CSV_PATH = os.getenv("WEATHER_CSV_PATH", DEFAULT_WEATHER_PATH)

# ==============================================================================
# THÔNG TIN 5 CAMERA TẠI HÀ NỘI
# ==============================================================================
CAMERAS = [
    {
        "camera_id": 1,
        "camera_name": "CAM_HN01",
        "location": "Nga tu Dai Co Viet - Giai Phong",
        "latitude": 21.0024,
        "longitude": 105.8432,
        "road_name": "Dai Co Viet",
        "number_of_lanes": 4,
    },
    {
        "camera_id": 2,
        "camera_name": "CAM_HN02",
        "location": "Nga tu Nguyen Trai - Khuat Duy Tien",
        "latitude": 20.9963,
        "longitude": 105.8005,
        "road_name": "Nguyen Trai",
        "number_of_lanes": 6,
    },
    {
        "camera_id": 3,
        "camera_name": "CAM_HN03",
        "location": "Cau Vinh Tuy",
        "latitude": 20.9978,
        "longitude": 105.8752,
        "road_name": "Cau Vinh Tuy",
        "number_of_lanes": 4,
    },
    {
        "camera_id": 4,
        "camera_name": "CAM_HN04",
        "location": "Nga tu Cau Giay",
        "latitude": 21.0285,
        "longitude": 105.7960,
        "road_name": "Cau Giay",
        "number_of_lanes": 4,
    },
    {
        "camera_id": 5,
        "camera_name": "CAM_HN05",
        "location": "Nga tu Tran Duy Hung - Hoang Minh Giam",
        "latitude": 21.0105,
        "longitude": 105.8015,
        "road_name": "Tran Duy Hung",
        "number_of_lanes": 6,
    },
]

# ==============================================================================
# CÁC PHÂN PHỐI VÀ HỆ SỐ / DISTRIBUTIONS & FACTORS
# ==============================================================================
# Phuong tien: car(28%), motorcycle(55%), bicycle(7%), truck(6%), bus(4%)
VEHICLE_TYPES = ["car", "motorcycle", "bicycle", "truck", "bus"]
VEHICLE_WEIGHTS = [0.28, 0.55, 0.07, 0.06, 0.04]

# Huong di chuyen
DIRECTIONS = ["north", "south", "east", "west"]

# Loai vi pham
VIOLATION_TYPES = [
    "RED_LIGHT",
    "CAR_IN_MOTORCYCLE_LANE",
    "FORBIDDEN_LINE",
    "WRONG_WAY",
]


def derive_weather_type(weather_code: int) -> str:
    """
    Quy doi WMO weather code sang weather_type:
    - 'clear': 0-3 (Clear sky, mainly clear, partly cloudy, overcast)
    - 'foggy': 45-48 (Fog)
    - 'drizzle': 51-55 (Drizzle light, moderate, dense)
    - 'rainy': 61-99 (Rain, rain showers, thunderstorm)
    """
    if 0 <= weather_code <= 3:
        return "clear"
    elif 45 <= weather_code <= 48:
        return "foggy"
    elif 51 <= weather_code <= 55:
        return "drizzle"
    elif 61 <= weather_code <= 99:
        return "rainy"
    else:
        return "clear"


def parse_weather_csv(filepath: str):
    """
    Doc file Open-Meteo CSV:
    - Dong 1-2: Metadata (latitude, longitude, v.v.)
    - Dong 3: Dong trong
    - Dong 4: Header
    - Dong 5 tro di: Du lieu 720 gio (Thang 9/2024)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Khong tim thay file thoi tiet Open-Meteo tai: {filepath}\n"
            "Hay dat file CSV Open-Meteo vao data/external/ hoac chay lai voi "
            "--weather-file <duong_dan> (hoac bien moi truong WEATHER_CSV_PATH)."
        )

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        # Bo qua 3 dong metadata dau tien
        for _ in range(3):
            next(f)

        reader = csv.DictReader(f)
        for row in reader:
            # Tim cot khong phu thuoc ky tu do (degree symbol encoding)
            time_str = row.get("time", "").strip()
            if not time_str:
                continue

            temp_val = None
            humid_val = None
            precip_val = None
            rain_val = None
            wcode_val = None
            wind_val = None

            for key, val in row.items():
                if not key:
                    continue
                k_lower = key.lower()
                if k_lower.startswith("temperature"):
                    temp_val = float(val) if val != "" else 28.0
                elif k_lower.startswith("relative_humidity"):
                    humid_val = float(val) if val != "" else 80.0
                elif k_lower.startswith("precipitation"):
                    precip_val = float(val) if val != "" else 0.0
                elif k_lower.startswith("rain"):
                    rain_val = float(val) if val != "" else 0.0
                elif k_lower.startswith("weather_code"):
                    wcode_val = int(val) if val != "" else 0
                elif k_lower.startswith("wind_speed"):
                    wind_val = float(val) if val != "" else 5.0

            # Chuan hoa timestamp: 2024-09-01T00:00 -> 2024-09-01 00:00:00
            try:
                dt = datetime.fromisoformat(time_str)
            except ValueError:
                dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M")
            formatted_ts = dt.strftime("%Y-%m-%d %H:%M:%S")

            rainfall = rain_val if rain_val is not None else (precip_val or 0.0)
            weather_type = derive_weather_type(wcode_val if wcode_val is not None else 0)

            records.append({
                "datetime": dt,
                "timestamp": formatted_ts,
                "temperature": round(temp_val if temp_val is not None else 28.0, 1),
                "humidity": round(humid_val if humid_val is not None else 80.0, 1),
                "rainfall": round(rainfall, 2),
                "weather_type": weather_type,
                "wind_speed": round(wind_val if wind_val is not None else 5.0, 1),
                "weather_code": wcode_val if wcode_val is not None else 0,
            })

    return records


def calculate_hourly_vehicle_count(dt: datetime, weather_type: str, rainfall: float, lanes: int) -> int:
    """
    Tinh so luong phuong tien trong 1 gio dua tren:
    - Gio trong ngay (rush hours: 7-9am va 5-7pm tang 2.5-3x)
    - Ngay trong tuan (cuoi tuan giam traffic gio cao diem)
    - Thoi tiet (mua giam traffic nhe, mua to giam nhieu hon)
    - So lan duong (4 lan vs 6 lan)
    """
    h = dt.hour
    is_weekend = dt.weekday() in (5, 6)

    # Base traffic theo khung gio
    if 0 <= h <= 4:
        base = random.randint(12, 22)
    elif h == 5:
        base = random.randint(28, 42)
    elif h == 6:
        base = random.randint(55, 75)
    elif h in (7, 8):  # Cao diem sang (7:00 - 8:59)
        base = random.randint(75, 95) * (1.1 if is_weekend else 2.65)
    elif h == 9:
        base = random.randint(75, 95) * (1.1 if is_weekend else 1.4)
    elif 10 <= h <= 15:
        base = random.randint(60, 85)
    elif h == 16:
        base = random.randint(75, 95) * (1.1 if is_weekend else 1.45)
    elif h in (17, 18):  # Cao diem chieu (17:00 - 18:59)
        base = random.randint(75, 95) * (1.2 if is_weekend else 2.85)
    elif h == 19:
        base = random.randint(75, 95) * (1.15 if is_weekend else 1.5)
    elif 20 <= h <= 21:
        base = random.randint(55, 75)
    else:  # 22, 23
        base = random.randint(30, 48)

    # He so thoi tiet
    if rainfall > 5.0 or weather_type == "rainy":
        weather_mult = random.uniform(0.72, 0.82)  # Mua to: giam 18-28%
    elif rainfall > 0.1 or weather_type == "drizzle":
        weather_mult = random.uniform(0.88, 0.95)  # Mua phun: giam 5-12%
    elif weather_type == "foggy":
        weather_mult = random.uniform(0.90, 0.96)
    else:
        weather_mult = 1.0

    # He so lan duong
    lane_mult = 1.15 if lanes >= 6 else 0.90

    total_count = int(base * weather_mult * lane_mult)
    return max(total_count, 5)


def generate_vehicle_speed(dt: datetime, vehicle_type: str, weather_type: str, rainfall: float) -> float:
    """
    Tinh toc do phuong tien (km/h):
    - Cao diem cham hon (~20-25 km/h)
    - Gio thuong (~35-45 km/h)
    - Mua giam toc do
    - Xe dap cham hon, xe may / oto binh thuong, xe buyt / tai giam nhe
    """
    h = dt.hour
    is_weekend = dt.weekday() in (5, 6)
    is_rush = h in (7, 8, 17, 18)

    # Base speed theo thoi gian
    if is_rush:
        base_speed = random.gauss(32.0 if is_weekend else 22.5, 3.5)
    elif h in (6, 9, 16, 19):
        base_speed = random.gauss(34.0 if is_weekend else 30.5, 4.0)
    elif 10 <= h <= 15 or 20 <= h <= 21:
        base_speed = random.gauss(39.0, 4.0)
    else:  # 22-5: Dem duong vang
        base_speed = random.gauss(46.0, 4.5)

    # Anh huong boi mua
    if rainfall > 5.0 or weather_type == "rainy":
        base_speed -= random.uniform(6.5, 10.5)
    elif rainfall > 0.1 or weather_type == "drizzle":
        base_speed -= random.uniform(2.5, 4.5)
    elif weather_type == "foggy":
        base_speed -= random.uniform(3.0, 5.0)

    # Dieu chinh theo loai xe
    if vehicle_type == "bicycle":
        # Xe dap co dai toc do rieng
        speed = random.uniform(11.0, 17.0)
        if rainfall > 0.1:
            speed -= random.uniform(1.0, 2.5)
        return round(max(speed, 5.0), 1)

    speed = base_speed
    if vehicle_type == "car":
        speed += random.uniform(0.0, 3.0)
    elif vehicle_type == "motorcycle":
        speed += random.uniform(-1.0, 2.0)
    elif vehicle_type == "bus":
        speed -= random.uniform(3.0, 6.0)
    elif vehicle_type == "truck":
        speed -= random.uniform(2.5, 5.5)

    # Gioi han toi thieu
    speed = max(speed, 6.0)
    return round(speed, 1)


def generate_violation(vehicle_type: str, weather_type: str, rainfall: float) -> str:
    """
    Xac dinh vi pham giao thong:
    - 85% None, 15% vi pham
    - Mua lam tang nhe ty le vi pham (troi mua khuat tam nhin, voi vang vuot den, lan lan)
    - Cac loai: RED_LIGHT, CAR_IN_MOTORCYCLE_LANE, FORBIDDEN_LINE, WRONG_WAY
    """
    violation_rate = 0.15
    if rainfall > 5.0 or weather_type == "rainy":
        violation_rate = 0.20
    elif rainfall > 0.1 or weather_type == "drizzle":
        violation_rate = 0.17
    elif weather_type == "foggy":
        violation_rate = 0.18

    if random.random() >= violation_rate:
        return ""

    if vehicle_type == "car":
        return random.choices(
            VIOLATION_TYPES,
            weights=[0.35, 0.35, 0.20, 0.10],
            k=1
        )[0]
    elif vehicle_type in ("truck", "bus"):
        return random.choices(
            VIOLATION_TYPES,
            weights=[0.30, 0.30, 0.30, 0.10],
            k=1
        )[0]
    else:  # motorcycle, bicycle
        return random.choices(
            ["RED_LIGHT", "FORBIDDEN_LINE", "WRONG_WAY"],
            weights=[0.45, 0.40, 0.15],
            k=1
        )[0]


def generate_all_data(weather_csv_path: str, output_dir: str, seed: int = 42):
    """
    Ham chinh thuc hien doc thoi tiet va sinh toan bo du lieu.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("TRAFFIC & WEATHER DATA GENERATOR - HA NOI")
    print("=" * 65)
    print(f"Weather Source CSV: {weather_csv_path}")
    print(f"Output Directory:   {output_dir}")
    print(f"Random Seed:        {seed}")
    print("=" * 65)

    # 1. Doc file thoi tiet
    print("\n[1/4] Dang doc du lieu thoi tiet tu Open-Meteo CSV...")
    weather_records = parse_weather_csv(weather_csv_path)
    total_hours = len(weather_records)
    print(f"  -> Da doc {total_hours} gio thoi tiet (Thang 9/2024: 2024-09-01 den 2024-09-30)")

    # 2. Ghi file cameras.csv
    cameras_file = os.path.join(output_dir, "cameras.csv")
    print(f"\n[2/4] Dang tao file camera: {cameras_file}")
    with open(cameras_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "camera_id", "camera_name", "location", "latitude",
            "longitude", "road_name", "number_of_lanes"
        ])
        writer.writeheader()
        for cam in CAMERAS:
            writer.writerow(cam)
    print(f"  -> Da tao {len(CAMERAS)} ban ghi camera")

    # 3. Ghi file weather_data.csv (da chuan hoa)
    weather_file = os.path.join(output_dir, "weather_data.csv")
    print(f"\n[3/4] Dang tao file thoi tiet chuan hoa: {weather_file}")
    weather_type_counts = {}
    with open(weather_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "temperature", "humidity", "rainfall",
            "weather_type", "wind_speed", "weather_code"
        ])
        writer.writeheader()
        for w in weather_records:
            w_out = {
                "timestamp": w["timestamp"],
                "temperature": w["temperature"],
                "humidity": w["humidity"],
                "rainfall": w["rainfall"],
                "weather_type": w["weather_type"],
                "wind_speed": w["wind_speed"],
                "weather_code": w["weather_code"],
            }
            writer.writerow(w_out)
            wt = w["weather_type"]
            weather_type_counts[wt] = weather_type_counts.get(wt, 0) + 1
    print(f"  -> Da tao {len(weather_records)} ban ghi thoi tiet")

    # 4. Sinh du lieu giao thong va vi pham
    traffic_file = os.path.join(output_dir, "traffic_data.csv")
    violations_file = os.path.join(output_dir, "violations.csv")
    print(f"\n[4/4] Dang sinh du lieu giao thong ({traffic_file}) & vi pham ({violations_file})...")

    total_traffic_rows = 0
    total_violations_rows = 0
    vehicle_counts = {vt: 0 for vt in VEHICLE_TYPES}
    violation_counts = {vt: 0 for vt in VIOLATION_TYPES}
    speed_sum = 0.0
    rush_speed_sum = 0.0
    rush_speed_count = 0
    rain_speed_sum = 0.0
    rain_speed_count = 0

    with open(traffic_file, "w", encoding="utf-8", newline="") as f_traffic, \
         open(violations_file, "w", encoding="utf-8", newline="") as f_violation:

        traffic_writer = csv.DictWriter(f_traffic, fieldnames=[
            "camera_id", "timestamp", "vehicle_type", "speed", "direction", "violation"
        ])
        traffic_writer.writeheader()

        violation_writer = csv.DictWriter(f_violation, fieldnames=[
            "camera_id", "timestamp", "vehicle_type", "violation_type", "speed"
        ])
        violation_writer.writeheader()

        # Duyet tung gio thoi tiet
        for hour_idx, w in enumerate(weather_records, 1):
            base_dt = w["datetime"]
            w_type = w["weather_type"]
            w_rain = w["rainfall"]
            is_rush = base_dt.hour in (7, 8, 17, 18) and (base_dt.weekday() < 5)
            is_raining = w_rain > 0.1 or w_type in ("drizzle", "rainy")

            # Sinh du lieu cho tung camera trong gio do
            hour_vehicles = []
            for cam in CAMERAS:
                cam_id = cam["camera_id"]
                num_vehicles = calculate_hourly_vehicle_count(
                    base_dt, w_type, w_rain, cam["number_of_lanes"]
                )

                for _ in range(num_vehicles):
                    # Random phut va giay trong gio
                    minute = random.randint(0, 59)
                    second = random.randint(0, 59)
                    vehicle_dt = base_dt + timedelta(minutes=minute, seconds=second)
                    ts_str = vehicle_dt.strftime("%Y-%m-%d %H:%M:%S")

                    v_type = random.choices(VEHICLE_TYPES, weights=VEHICLE_WEIGHTS, k=1)[0]
                    v_speed = generate_vehicle_speed(base_dt, v_type, w_type, w_rain)
                    v_dir = random.choice(DIRECTIONS)
                    v_violation = generate_violation(v_type, w_type, w_rain)

                    hour_vehicles.append({
                        "vehicle_dt": vehicle_dt,
                        "camera_id": cam_id,
                        "timestamp": ts_str,
                        "vehicle_type": v_type,
                        "speed": v_speed,
                        "direction": v_dir,
                        "violation": v_violation,
                        "is_rush": is_rush,
                        "is_raining": is_raining,
                    })

            # Sap xep phuong tien trong gio theo thoi gian de dam bao tinh tuan tu thoi gian
            hour_vehicles.sort(key=lambda x: x["vehicle_dt"])

            for v in hour_vehicles:
                # Ghi vao traffic_data.csv
                traffic_writer.writerow({
                    "camera_id": v["camera_id"],
                    "timestamp": v["timestamp"],
                    "vehicle_type": v["vehicle_type"],
                    "speed": v["speed"],
                    "direction": v["direction"],
                    "violation": v["violation"],
                })

                total_traffic_rows += 1
                vehicle_counts[v["vehicle_type"]] += 1
                speed_sum += v["speed"]

                if v["is_rush"]:
                    rush_speed_sum += v["speed"]
                    rush_speed_count += 1

                if v["is_raining"]:
                    rain_speed_sum += v["speed"]
                    rain_speed_count += 1

                # Neu co vi pham -> ghi vao violations.csv
                if v["violation"]:
                    violation_writer.writerow({
                        "camera_id": v["camera_id"],
                        "timestamp": v["timestamp"],
                        "vehicle_type": v["vehicle_type"],
                        "violation_type": v["violation"],
                        "speed": v["speed"],
                    })
                    total_violations_rows += 1
                    violation_counts[v["violation"]] += 1

            if hour_idx % 100 == 0 or hour_idx == total_hours:
                pct = (hour_idx / total_hours) * 100
                print(
                    f"  [Tien do] Da xu ly {hour_idx:3d}/{total_hours} gio ({pct:5.1f}%) "
                    f"-> {total_traffic_rows:,} xe, {total_violations_rows:,} vi pham"
                )

    # ==============================================================================
    # THỐNG KÊ CHI TIẾT SAU KHI SINH DỮ LIỆU
    # ==============================================================================
    print("\n" + "=" * 65)
    print("THONG KE HOAN TAT (FINAL STATISTICS)")
    print("=" * 65)
    print(f"Tong so gio thoi tiet da xu ly: {total_hours:,} gio")
    print(f"Tong so camera:                   {len(CAMERAS)}")
    print(f"Tong so ban ghi traffic_data:     {total_traffic_rows:,} dong")
    print(f"Tong so ban ghi violations:       {total_violations_rows:,} dong "
          f"({(total_violations_rows / total_traffic_rows) * 100:.2f}%)")

    print("\n--- Co cau loai phuong tien ---")
    for vt in VEHICLE_TYPES:
        c = vehicle_counts[vt]
        print(f"  * {vt:12s}: {c:8,d} ({c / total_traffic_rows * 100:5.2f}%)")

    print("\n--- Co cau vi pham giao thong ---")
    for vt in VIOLATION_TYPES:
        c = violation_counts[vt]
        pct = (c / total_violations_rows * 100) if total_violations_rows > 0 else 0
        print(f"  * {vt:24s}: {c:6,d} ({pct:5.2f}%)")

    print("\n--- Thong ke toc do trung binh ---")
    avg_speed = speed_sum / total_traffic_rows if total_traffic_rows > 0 else 0
    avg_rush_speed = rush_speed_sum / rush_speed_count if rush_speed_count > 0 else 0
    avg_rain_speed = rain_speed_sum / rain_speed_count if rain_speed_count > 0 else 0
    print(f"  * Toc do TB toan bo:          {avg_speed:.2f} km/h")
    print(f"  * Toc do TB gio cao diem:     {avg_rush_speed:.2f} km/h")
    print(f"  * Toc do TB khi troi mua:     {avg_rain_speed:.2f} km/h")

    print("\n--- Co cau thoi tiet theo gio (weather_data) ---")
    for wt, c in sorted(weather_type_counts.items()):
        print(f"  * {wt:10s}: {c:4d} gio ({c / total_hours * 100:5.2f}%)")

    print("\n--- File dau ra da tao ---")
    out_files = [
        ("traffic_data.csv", traffic_file, total_traffic_rows),
        ("weather_data.csv", weather_file, len(weather_records)),
        ("cameras.csv", cameras_file, len(CAMERAS)),
        ("violations.csv", violations_file, total_violations_rows),
    ]
    for name, path, rows in out_files:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  * {name:18s}: {size_mb:6.2f} MB ({rows:,} rows) -> {path}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Tao du lieu giao thong Ha Noi dua tren du lieu thoi tiet Open-Meteo."
    )
    parser.add_argument(
        "--weather-file",
        default=WEATHER_CSV_PATH,
        help=f"Duong dan file thoi tiet Open-Meteo CSV (Mac dinh: {WEATHER_CSV_PATH})"
    )
    parser.add_argument(
        "--output-dir",
        default=DATA_RAW_DIR,
        help=f"Thu muc luu file CSV ket qua (Mac dinh: {DATA_RAW_DIR})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed cho tinh tai lap (Mac dinh: 42)"
    )

    args = parser.parse_args()
    generate_all_data(
        weather_csv_path=args.weather_file,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
