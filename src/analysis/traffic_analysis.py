"""
Phan tich va truc quan hoa du lieu giao thong do thi Ha Noi.
Doc du lieu truc tiep tu cac file CSV (data/processed/ va data/raw/)
va xuat 6 bieu do chat luong cao cung bao cao tong hop ra console.
Khong yeu cau ket noi PostgreSQL.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CAU HINH DUONG DAN VA THIET LAP MATPLOTLIB
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
CHARTS_DIR = os.path.join(DATA_PROCESSED, "charts")

os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# Thiet lap phong cach do hoa sach se, chuyen nghiep
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
plt.rcParams["axes.edgecolor"] = "#bdc3c7"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#ecf0f1"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.8
plt.rcParams["figure.autolayout"] = True

# Thong tin cac camera quan sat tai Ha Noi
CAMERA_NAMES = {
    1: {"code": "CAM_HN01", "name": "Nga tu Dai Co Viet - Giai Phong", "lanes": 4},
    2: {"code": "CAM_HN02", "name": "Nga tu Nguyen Trai - Khuat Duy Tien", "lanes": 6},
    3: {"code": "CAM_HN03", "name": "Cau Vinh Tuy", "lanes": 4},
    4: {"code": "CAM_HN04", "name": "Nga tu Cau Giay", "lanes": 4},
    5: {"code": "CAM_HN05", "name": "Nga tu Tran Duy Hung - Hoang Minh Giam", "lanes": 6},
    "CAM_HN01": {"code": "CAM_HN01", "name": "Nga tu Dai Co Viet - Giai Phong", "lanes": 4},
    "CAM_HN02": {"code": "CAM_HN02", "name": "Nga tu Nguyen Trai - Khuat Duy Tien", "lanes": 6},
    "CAM_HN03": {"code": "CAM_HN03", "name": "Cau Vinh Tuy", "lanes": 4},
    "CAM_HN04": {"code": "CAM_HN04", "name": "Nga tu Cau Giay", "lanes": 4},
    "CAM_HN05": {"code": "CAM_HN05", "name": "Nga tu Tran Duy Hung - Hoang Minh Giam", "lanes": 6},
}


# ==============================================================================
# 2. KIEM TRA DU LIEU DAU VAO
# ==============================================================================
def ensure_sample_data(traffic_csv, violations_csv, weather_csv, cameras_csv):
    """
    Kiem tra du lieu dau vao da ton tai.

    Truoc day ham nay tu dong sinh mot bo du lieu mau KHAC (7 ngay, loai vi pham
    khac) khi thieu file, lam bieu do khong khop voi du lieu that cua du an.
    Nay bao loi ro rang de nguoi dung chay dung pipeline.
    """
    required = {
        "traffic_summary.csv (Spark)": traffic_csv,
        "violations.csv": violations_csv,
        "weather_data.csv": weather_csv,
    }
    missing = [name for name, path in required.items() if not os.path.exists(path)]
    if missing:
        print("[LOI] Thieu du lieu dau vao: " + ", ".join(missing))
        print("Hay chay theo thu tu:")
        print("  1) python src/ingestion/generate_data.py   (tao data/raw/*.csv)")
        print("  2) python src/processing/spark_batch.py    (tao data/processed/traffic_summary.csv)")
        raise FileNotFoundError("Thieu file dau vao: " + ", ".join(missing))


def weighted_speed(df, by=None):
    """
    Toc do trung binh co trong so theo so xe (total_vehicles).
    Trung binh thuong cua avg_speed coi 1 gio dem co 12 xe ngang 1 gio cao diem
    co 3.000 xe, nen bi lech cao (gio vang xe chay nhanh).
    """
    tmp = df.assign(_sv=df["avg_speed"] * df["total_vehicles"])
    if by is None:
        return tmp["_sv"].sum() / tmp["total_vehicles"].sum()
    g = tmp.groupby(by).agg(_sv=("_sv", "sum"), _v=("total_vehicles", "sum"))
    return g["_sv"] / g["_v"]


# ==============================================================================
# 3. HAM DOC VA TIEN XU LY DU LIEU
# ==============================================================================
def load_datasets():
    """Doc du lieu tu 3 file CSV: traffic_summary, violations, weather_data."""
    traffic_csv = os.path.join(DATA_PROCESSED, "traffic_summary.csv")
    violations_csv = os.path.join(DATA_RAW, "violations.csv")
    weather_csv = os.path.join(DATA_RAW, "weather_data.csv")
    cameras_csv = os.path.join(DATA_RAW, "cameras.csv")

    ensure_sample_data(traffic_csv, violations_csv, weather_csv, cameras_csv)

    print("\nDang doc du lieu tu CSV...")
    df_summary = pd.read_csv(traffic_csv)
    df_violations = pd.read_csv(violations_csv)
    df_weather = pd.read_csv(weather_csv)

    # Chuan hoa kieu du lieu
    df_summary["summary_date"] = pd.to_datetime(df_summary["summary_date"])
    df_summary["hour_of_day"] = df_summary["hour_of_day"].astype(int)
    df_summary["is_weekend"] = df_summary["summary_date"].dt.dayofweek.isin([5, 6])
    df_summary["day_type"] = df_summary["is_weekend"].map({True: "Cuoi tuan", False: "Ngay thuong"})

    print(f"  - traffic_summary: {len(df_summary):,} ban ghi")
    print(f"  - violations:      {len(df_violations):,} ban ghi")
    print(f"  - weather_data:    {len(df_weather):,} ban ghi")

    return df_summary, df_violations, df_weather


# ==============================================================================
# 4. CAC HAM VE BIEU DO (6 CHARTS)
# ==============================================================================

def chart_01_peak_hours(df_summary):
    """
    Chart 1: Bar chart luu luong phuong tien theo gio (0-23h).
    Noi bat gio cao diem sang (7-9h) va chieu (17-19h).
    """
    hourly = df_summary.groupby("hour_of_day")["total_vehicles"].sum().reset_index()

    plt.figure(figsize=(11, 5.8), dpi=300)
    ax = plt.gca()

    # Dinh nghia mau sac: do cam cho cao diem, xanh nuoc bien cho binh thuong
    colors = []
    for h in hourly["hour_of_day"]:
        if h in [7, 8, 9, 17, 18, 19]:
            colors.append("#d63031")  # Mau do cho gio cao diem
        elif 10 <= h <= 16:
            colors.append("#0984e3")  # Mau xanh cho gio lam viec
        else:
            colors.append("#74b9ff")  # Mau xanh nhat cho ban dem/sang som

    bars = ax.bar(hourly["hour_of_day"], hourly["total_vehicles"], color=colors, width=0.72, edgecolor="#2d3436", linewidth=0.6)

    # Hien thi so lieu tren tung cot
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + (hourly["total_vehicles"].max() * 0.012),
                f"{int(h/1000):,}k" if h >= 10000 else f"{int(h):,}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#2d3436")

    # Tieu de va chu thich
    plt.title("Luu Luong Phuong Tien Theo Khung Gio Trong Ngay Tai Ha Noi\n(Peak Hours Analysis - 24 Hours Traffic Volume)",
              fontsize=13, fontweight="bold", pad=14, color="#2c3e50")
    plt.xlabel("Khung gio (Gio trong ngay: 0h - 23h)", fontsize=10.5, labelpad=8)
    plt.ylabel("Tong luong phuong tien (Xe)", fontsize=10.5, labelpad=8)
    plt.xticks(range(0, 24), [f"{h}h" for h in range(24)], fontsize=9)
    plt.yticks(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))

    # Chu thich cao diem
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d63031", edgecolor="#2d3436", label="Gio cao diem (07h-09h & 17h-19h)"),
        Patch(facecolor="#0984e3", edgecolor="#2d3436", label="Khung gio binh thuong (10h-16h)"),
        Patch(facecolor="#74b9ff", edgecolor="#2d3436", label="Khung gio dem / sang som (20h-06h)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, framealpha=0.9, fontsize=9)

    ax.set_ylim(0, hourly["total_vehicles"].max() * 1.12)
    chart_path = os.path.join(CHARTS_DIR, "chart_01_peak_hours.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"  [1/6] Da xuat: {chart_path}")


def chart_02_speed_by_hour(df_summary):
    """
    Chart 2: Line chart toc do trung binh theo gio (0-23h)
    Kem theo cac nguong phan loai un tac (critical < 15, high < 25, low >= 35 km/h).
    """
    hourly_speed = weighted_speed(df_summary, "hour_of_day").rename("avg_speed").reset_index()

    plt.figure(figsize=(11, 5.8), dpi=300)
    ax = plt.gca()

    x = hourly_speed["hour_of_day"]
    y = hourly_speed["avg_speed"]

    # Duong bieu dien toc do
    ax.plot(x, y, color="#e17055", linewidth=2.8, marker="o", markersize=6,
            markerfacecolor="#d63031", markeredgecolor="#ffffff", markeredgewidth=1.5,
            label="Toc do trung binh (km/h)", zorder=4)

    # To mau vung ben duoi
    ax.fill_between(x, y, color="#fab1a0", alpha=0.35, zorder=2)

    # Cac nguong un tac theo tieu chuan giao thong do thi
    ax.axhline(15, color="#d63031", linestyle="--", linewidth=1.2, alpha=0.8,
               label="Nguong un tac nghiem trong (Critical < 15 km/h)", zorder=3)
    ax.axhline(25, color="#e67e22", linestyle="--", linewidth=1.2, alpha=0.8,
               label="Nguong un tac cao (High < 25 km/h)", zorder=3)
    ax.axhline(35, color="#00b894", linestyle="--", linewidth=1.2, alpha=0.8,
               label="Nguong luu thong tot (Low >= 35 km/h)", zorder=3)

    # Danh dau cac diem thap nhat (dinh diem un tac)
    min_idx = y.idxmin()
    min_h = x[min_idx]
    min_spd = y[min_idx]
    ax.annotate(f"Diem thap nhat: {min_spd:.1f} km/h ({min_h}h)",
                xy=(min_h, min_spd), xytext=(min_h + 0.8, min_spd - 4.5),
                arrowprops=dict(facecolor="#d63031", shrink=0.08, width=1.5, headwidth=6),
                fontweight="bold", fontsize=9, color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#d63031", alpha=0.9))

    plt.title("Dien Bien Toc Do Trung Binh Theo Gio Trong Ngay Tai Ha Noi\n(Hourly Average Speed & Congestion Thresholds)",
              fontsize=13, fontweight="bold", pad=14, color="#2c3e50")
    plt.xlabel("Gio trong ngay (0h - 23h)", fontsize=10.5, labelpad=8)
    plt.ylabel("Toc do trung binh (km/h)", fontsize=10.5, labelpad=8)
    plt.xticks(range(0, 24), [f"{h}h" for h in range(24)], fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylim(0, max(50, y.max() + 8))
    ax.legend(loc="lower center", frameon=True, framealpha=0.92, fontsize=8.8)

    chart_path = os.path.join(CHARTS_DIR, "chart_02_speed_by_hour.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"  [2/6] Da xuat: {chart_path}")


def chart_03_vehicle_types(df_summary):
    """
    Chart 3: Pie chart / Donut chart ve co cau loai phuong tien.
    Phan anh thuc te Ha Noi voi xe may chiem ap dao.
    """
    totals = {
        "Xe may (Motorcycle)": df_summary["motorcycle_count"].sum(),
        "O to con (Car)": df_summary["car_count"].sum(),
        "Xe bus (Bus)": df_summary["bus_count"].sum(),
        "Xe tai (Truck)": df_summary["truck_count"].sum(),
        "Xe dap (Bicycle)": df_summary["bicycle_count"].sum(),
    }

    labels = list(totals.keys())
    values = list(totals.values())
    total_val = sum(values)

    colors = ["#0984e3", "#e17055", "#00b894", "#fdcb6e", "#6c5ce7"]
    explode = (0.05, 0.03, 0, 0, 0)  # Lam noi bat xe may va o to

    plt.figure(figsize=(9, 7), dpi=300)
    ax = plt.gca()

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct="%1.1f%%",
        pctdistance=0.78,
        explode=explode,
        colors=colors,
        startangle=140,
        wedgeprops=dict(width=0.48, edgecolor="white", linewidth=2),
    )

    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")
        autotext.set_color("#2d3436")

    # Tam vong tron giua tao hieu ung donut
    ax.text(0, 0, f"Ha Noi Traffic\nTong cong:\n{total_val:,} xe",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#2c3e50")

    # Legend kem theo so luong tuyet doi
    legend_labels = [f"{label}: {val:,} xe ({val/total_val*100:.1f}%)" for label, val in zip(labels, values)]
    ax.legend(wedges, legend_labels, title="Loai phuong tien", loc="center left",
              bbox_to_anchor=(0.88, 0.5), fontsize=9, title_fontsize=10, frameon=True)

    plt.title("Co Cau Cac Loai Phuong Tien Tham Gia Giao Thong Tai Ha Noi\n(Vehicle Type Distribution)",
              fontsize=13, fontweight="bold", pad=15, color="#2c3e50")

    chart_path = os.path.join(CHARTS_DIR, "chart_03_vehicle_types.png")
    plt.tight_layout()
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    print(f"  [3/6] Da xuat: {chart_path}")


def chart_04_weather_impact(df_summary):
    """
    Chart 4: Grouped bar chart so sanh avg_speed va vehicle_count qua cac kieu thoi tiet
    (clear, drizzle, rainy). Su dung truc kep (twinx) de truc quan hoa ca 2 thang do.
    """
    weather_labels_map = {
        "clear": "Troi quang (Clear)",
        "drizzle": "Mua phuon (Drizzle)",
        "rainy": "Mua rao (Rainy)",
    }

    # Gom nhom theo thoi tiet
    weather_grp = df_summary.groupby("weather_type").agg(
        avg_vehicles=("total_vehicles", "mean"),
        total_vehicles=("total_vehicles", "sum")
    ).reset_index()
    weather_grp["avg_speed"] = weather_grp["weather_type"].map(
        weighted_speed(df_summary, "weather_type")
    )

    # Dam bao thu tu logic: clear -> drizzle -> rainy
    order = ["clear", "drizzle", "rainy"]
    weather_grp["order"] = weather_grp["weather_type"].map(lambda x: order.index(x) if x in order else 99)
    weather_grp = weather_grp.sort_values("order").reset_index(drop=True)

    x_labels = [weather_labels_map.get(w, w) for w in weather_grp["weather_type"]]
    x = np.arange(len(x_labels))
    bar_width = 0.32

    fig, ax1 = plt.subplots(figsize=(10, 5.8), dpi=300)
    ax2 = ax1.twinx()

    # Bar 1: Toc do trung binh (km/h) tren ax1 (mau xanh)
    rects1 = ax1.bar(x - bar_width / 2, weather_grp["avg_speed"], width=bar_width,
                     color="#0984e3", label="Toc do trung binh (km/h)", edgecolor="#2d3436", linewidth=0.6)

    # Bar 2: Luong xe trung binh / gio tren ax2 (mau cam)
    rects2 = ax2.bar(x + bar_width / 2, weather_grp["avg_vehicles"], width=bar_width,
                     color="#e17055", label="Luu luong trung binh (xe/gio)", edgecolor="#2d3436", linewidth=0.6)

    # Nhan gia tri tren tung cot
    for rect in rects1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2.0, h + 0.6,
                 f"{h:.1f} km/h", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0984e3")

    for rect in rects2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2.0, h + weather_grp["avg_vehicles"].max() * 0.02,
                 f"{int(h):,} xe/h", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#d63031")

    # Dinh dang truc
    ax1.set_xlabel("Dieu kien thoi tiet", fontsize=10.5, labelpad=8)
    ax1.set_ylabel("Toc do trung binh (km/h)", fontsize=10.5, color="#0984e3", labelpad=8)
    ax2.set_ylabel("Luu luong trung binh / gio (xe/gio)", fontsize=10.5, color="#e17055", labelpad=8)
    ax1.tick_params(axis="y", labelcolor="#0984e3")
    ax2.tick_params(axis="y", labelcolor="#e17055")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=10, fontweight="bold")

    ax1.set_ylim(0, max(weather_grp["avg_speed"]) * 1.25)
    ax2.set_ylim(0, max(weather_grp["avg_vehicles"]) * 1.25)
    ax2.grid(False)  # Tat grid ax2 de khong bi chong cheo

    plt.title("Anh Huong Cua Thoi Tiet Den Toc Do Va Luu Luong Giao Thong Ha Noi\n(Weather Impact: Average Speed vs Traffic Volume)",
              fontsize=13, fontweight="bold", pad=15, color="#2c3e50")

    # Gop 2 legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, framealpha=0.92, fontsize=9.5)

    chart_path = os.path.join(CHARTS_DIR, "chart_04_weather_impact.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"  [4/6] Da xuat: {chart_path}")


def chart_05_violations(df_violations):
    """
    Chart 5: Horizontal bar chart thong ke cac loai vi pham giao thong.
    Sap xep theo tan suat giam dan.
    """
    col_name = "violation_type" if "violation_type" in df_violations.columns else "violation"
    viol_counts = df_violations[col_name].dropna().value_counts().reset_index()
    viol_counts.columns = ["violation_type", "count"]

    # Loai bo gia tri rong
    viol_counts = viol_counts[viol_counts["violation_type"].str.strip() != ""]
    # Sap xep tang dan de khi ve barh thi loai nhieu nhat nam o tren cung
    viol_counts = viol_counts.sort_values("count", ascending=True).reset_index(drop=True)

    total_viols = viol_counts["count"].sum()
    viol_counts["pct"] = viol_counts["count"] / total_viols * 100.0

    plt.figure(figsize=(10.5, 5.8), dpi=300)
    ax = plt.gca()

    # Mau sac chuyen tu nhat sang dam dan
    palette = ["#74b9ff", "#0984e3", "#fdcb6e", "#e17055", "#d63031", "#c0392b"]
    colors = palette[-len(viol_counts):] if len(viol_counts) <= len(palette) else ["#d63031"] * len(viol_counts)

    bars = ax.barh(viol_counts["violation_type"], viol_counts["count"],
                   color=colors, height=0.6, edgecolor="#2d3436", linewidth=0.6)

    # Hien thi so luong va ti le phan tram ben canh moi bar
    for bar, pct in zip(bars, viol_counts["pct"]):
        w = bar.get_width()
        ax.text(w + (viol_counts["count"].max() * 0.015), bar.get_y() + bar.get_height() / 2.0,
                f"{int(w):,} ({pct:.1f}%)",
                ha="left", va="center", fontsize=9.5, fontweight="bold", color="#2c3e50")

    plt.title("Thong Ke Cac Loai Vi Pham Giao Thong Pho Bien Tai Ha Noi\n(Traffic Violations by Category - Frequency & Percentage)",
              fontsize=13, fontweight="bold", pad=15, color="#2c3e50")
    plt.xlabel("So truong hop vi pham (Luot)", fontsize=10.5, labelpad=8)
    plt.ylabel("Hanh vi vi pham", fontsize=10.5, labelpad=8)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9.5, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))
    ax.set_xlim(0, viol_counts["count"].max() * 1.22)

    chart_path = os.path.join(CHARTS_DIR, "chart_05_violations.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"  [5/6] Da xuat: {chart_path}")


def chart_06_weekday_weekend(df_summary):
    """
    Chart 6: Grouped bar chart so sanh luu luong giao thong theo gio giua
    Ngay thuong (Weekday: Thu 2 - Thu 6) va Cuoi tuan (Weekend: Thu 7 - Chu Nhat).
    """
    # Tinh luu luong trung binh moi gio cho ngay thuong vs cuoi tuan
    pivot = df_summary.groupby(["hour_of_day", "day_type"])["total_vehicles"].mean().unstack().reset_index()

    # Dam bao co du 2 cot
    if "Ngay thuong" not in pivot:
        pivot["Ngay thuong"] = 0
    if "Cuoi tuan" not in pivot:
        pivot["Cuoi tuan"] = 0

    x = pivot["hour_of_day"]
    width = 0.38

    plt.figure(figsize=(11.5, 6), dpi=300)
    ax = plt.gca()

    rects_weekday = ax.bar(x - width / 2, pivot["Ngay thuong"], width,
                           label="Ngay thuong (Thu 2 - Thu 6)", color="#0984e3",
                           edgecolor="#2d3436", linewidth=0.5)
    rects_weekend = ax.bar(x + width / 2, pivot["Cuoi tuan"], width,
                           label="Cuoi tuan (Thu 7 - Chu Nhat)", color="#fdcb6e",
                           edgecolor="#2d3436", linewidth=0.5)

    # Danh dau cac dinh gio cao diem ngay thuong
    ax.annotate("Dinh sang\n(07h-08h)", xy=(7.8, pivot.loc[pivot["hour_of_day"] == 8, "Ngay thuong"].values[0]),
                xytext=(6.5, pivot["Ngay thuong"].max() * 0.98),
                arrowprops=dict(facecolor="#d63031", shrink=0.08, width=1.2, headwidth=5),
                fontsize=8.5, fontweight="bold", color="#c0392b", ha="center")

    ax.annotate("Dinh chieu\n(17h-18h)", xy=(17.8, pivot.loc[pivot["hour_of_day"] == 18, "Ngay thuong"].values[0]),
                xytext=(18.5, pivot["Ngay thuong"].max() * 0.98),
                arrowprops=dict(facecolor="#d63031", shrink=0.08, width=1.2, headwidth=5),
                fontsize=8.5, fontweight="bold", color="#c0392b", ha="center")

    plt.title("So Sanh Luu Luong Giao Thong: Ngay Thuong vs Cuoi Tuan Tai Ha Noi\n(Hourly Traffic Comparison: Weekday vs Weekend Volume)",
              fontsize=13, fontweight="bold", pad=15, color="#2c3e50")
    plt.xlabel("Gio trong ngay (0h - 23h)", fontsize=10.5, labelpad=8)
    plt.ylabel("Luu luong trung binh moi camera (Xe / Gio)", fontsize=10.5, labelpad=8)
    plt.xticks(range(0, 24), [f"{h}h" for h in range(24)], fontsize=9)
    plt.yticks(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"{int(val):,}"))
    ax.set_ylim(0, max(pivot["Ngay thuong"].max(), pivot["Cuoi tuan"].max()) * 1.15)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9.5)

    chart_path = os.path.join(CHARTS_DIR, "chart_06_weekday_weekend.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"  [6/6] Da xuat: {chart_path}")


# ==============================================================================
# 5. HAM IN BAO CAO TONG HOP VA KET LUAN PHAN TICH (TEXT SUMMARY)
# ==============================================================================
def print_text_summary(df_summary, df_violations, df_weather):
    """Xuat bao cao phan tich chuyen sau ra man hinh console."""
    total_vehicles = df_summary["total_vehicles"].sum()
    avg_speed_overall = weighted_speed(df_summary)
    total_violations = len(df_violations)

    # 1. Khung gio dong nhat
    hourly_vol = df_summary.groupby("hour_of_day")["total_vehicles"].sum()
    peak_hour = hourly_vol.idxmax()
    lowest_hour = hourly_vol.idxmin()

    # 2. Khung gio toc do thap nhat
    hourly_spd = weighted_speed(df_summary, "hour_of_day")
    slowest_hour = hourly_spd.idxmin()

    # 3. Khu vuc un tac nhat
    cam_summary = df_summary.groupby("camera_id").agg(
        total_veh=("total_vehicles", "sum"),
        critical_hours=("congestion_level", lambda s: (s.isin(["high", "critical"])).sum())
    ).reset_index()
    cam_summary["avg_spd"] = cam_summary["camera_id"].map(weighted_speed(df_summary, "camera_id"))

    # Map ten camera
    cam_summary["cam_name"] = cam_summary["camera_id"].map(
        lambda cid: CAMERA_NAMES.get(cid, {}).get("name", f"Camera {cid}")
    )
    cam_summary = cam_summary.sort_values("critical_hours", ascending=False)
    worst_cam = cam_summary.iloc[0]

    # 4. Co cau phuong tien
    motos = df_summary["motorcycle_count"].sum()
    cars = df_summary["car_count"].sum()
    buses = df_summary["bus_count"].sum()
    trucks = df_summary["truck_count"].sum()
    bikes = df_summary["bicycle_count"].sum()
    sum_veh = motos + cars + buses + trucks + bikes

    # 5. Thoi tiet
    weather_stats = df_summary.groupby("weather_type").agg(
        avg_veh=("total_vehicles", "mean")
    )
    weather_stats["avg_spd"] = weighted_speed(df_summary, "weather_type")

    # 6. Vi pham pho bien
    col_v = "violation_type" if "violation_type" in df_violations.columns else "violation"
    top_v = df_violations[col_v].dropna().value_counts()

    # 7. Ngay thuong vs Cuoi tuan
    day_stats = df_summary.groupby("day_type").agg(
        avg_veh=("total_vehicles", "mean"),
        viols=("violation_count", "sum")
    )
    day_stats["avg_spd"] = weighted_speed(df_summary, "day_type")

    print("\n" + "=" * 80)
    print("      BAO CAO PHAN TICH DU LIEU GIAO THONG DO THI HA NOI (BIG DATA SUMMARY)")
    print("=" * 80)

    print("\n1. TONG QUAN VE HE THONG DU LIEU (DATASET OVERVIEW):")
    print(f"   - Tong luong phuong tien ghi nhan:   {total_vehicles:,} luot xe")
    print(f"   - Toc do trung binh toan mang luoi:  {avg_speed_overall:.1f} km/h")
    print(f"   - Tong so vi pham da phat hien:      {total_violations:,} truong hop")
    print(f"   - So luong nut giao giam sat:        {df_summary['camera_id'].nunique()} vi tri trong diem tai Ha Noi")

    print("\n2. PHAN TICH KHUNG GIO CAO DIEM VA TOC DO (PEAK HOURS & SPEED):")
    print(f"   - Gio dong xe nhat trong ngay:       {peak_hour:02d}:00 ({hourly_vol.max():,} xe)")
    print(f"   - Gio vang xe nhat:                  {lowest_hour:02d}:00 ({hourly_vol.min():,} xe)")
    print(f"   - Gio co toc do luu thong cham nhat: {slowest_hour:02d}:00 ({hourly_spd.min():.1f} km/h)")
    # Nhan xet dua tren so lieu (khong dung ket luan viet san)
    top_hours = hourly_vol.sort_values(ascending=False).head(4).index.sort_values().tolist()
    top_share = hourly_vol.loc[top_hours].sum() / hourly_vol.sum() * 100
    print("   - Nhan xet (tinh tu du lieu):")
    print(f"       + 4 gio dong xe nhat: {', '.join(f'{h:02d}h' for h in top_hours)} "
          f"(chiem {top_share:.1f}% luu luong ca ngay)")
    print(f"       + Toc do thap nhat luc {slowest_hour:02d}h: {hourly_spd.min():.1f} km/h, "
          f"cao nhat luc {hourly_spd.idxmax():02d}h: {hourly_spd.max():.1f} km/h")

    print("\n3. DIEM NONG UN TAC GIAO THONG (BOTTLENECK LOCATIONS):")
    for idx, row in cam_summary.iterrows():
        print(f"   * Camera {row['camera_id']} - {row['cam_name']}:")
        print(f"       + So gio un tac/nghiem trong: {row['critical_hours']} gio")
        print(f"       + Toc do trung binh:         {row['avg_spd']:.1f} km/h")
        print(f"       + Tong luu luong:            {row['total_veh']:,} xe")
    print(f"   -> Nut giao un tac nghiem trong nhat: {worst_cam['cam_name']}")

    print("\n4. CO CAU PHUONG TIEN GIAO THONG (VEHICLE BREAKDOWN):")
    print(f"   - Xe may (Motorcycle): {motos:,} xe ({motos/sum_veh*100:.1f}%) -> Phuong tien chu dao tuyet doi")
    print(f"   - O to con (Car):      {cars:,} xe ({cars/sum_veh*100:.1f}%)")
    print(f"   - Xe bus (Bus):        {buses:,} xe ({buses/sum_veh*100:.1f}%)")
    print(f"   - Xe tai (Truck):      {trucks:,} xe ({trucks/sum_veh*100:.1f}%)")
    print(f"   - Xe dap (Bicycle):    {bikes:,} xe ({bikes/sum_veh*100:.1f}%)")

    print("\n5. TAC DONG CUA THOI TIET DEN GIAO THONG (WEATHER IMPACT):")
    if "clear" in weather_stats.index and "rainy" in weather_stats.index:
        spd_drop = (weather_stats.loc["clear", "avg_spd"] - weather_stats.loc["rainy", "avg_spd"]) / weather_stats.loc["clear", "avg_spd"] * 100
        print(f"   - Troi quang (Clear):  Toc do {weather_stats.loc['clear', 'avg_spd']:.1f} km/h | Luu luong {weather_stats.loc['clear', 'avg_veh']:.0f} xe/h")
        if "drizzle" in weather_stats.index:
            print(f"   - Mua phun (Drizzle):  Toc do {weather_stats.loc['drizzle', 'avg_spd']:.1f} km/h | Luu luong {weather_stats.loc['drizzle', 'avg_veh']:.0f} xe/h")
        print(f"   - Mua rao (Rainy):     Toc do {weather_stats.loc['rainy', 'avg_spd']:.1f} km/h | Luu luong {weather_stats.loc['rainy', 'avg_veh']:.0f} xe/h")
        print(f"   -> Mua lon lam giam toc do di chuyen trung binh khoang {spd_drop:.1f}% va gia tang nguy co un tac keo dai.")

    print("\n6. PHAN TICH CAC LOAI VI PHAM PHO BIEN (VIOLATIONS BREAKDOWN):")
    for v_name, v_cnt in top_v.head(5).items():
        print(f"   - {v_name:<25}: {v_cnt:>6,} truong hop ({v_cnt/total_violations*100:>4.1f}%)")

    print("\n7. SO SANH NGAY THUONG VS CUOI TUAN (WEEKDAY VS WEEKEND):")
    for d_type, row in day_stats.iterrows():
        print(f"   - {d_type:<12}: Luu luong TB = {row['avg_veh']:.0f} xe/h | Toc do TB = {row['avg_spd']:.1f} km/h | Tong vi pham = {int(row['viols']):,}")
    # So sanh khung 19h-22h giua cuoi tuan va ngay thuong (tinh that, khong viet san)
    evening = df_summary[df_summary["hour_of_day"].between(19, 22)]
    ev = evening.groupby("day_type")["total_vehicles"].mean()
    if "Cuoi tuan" in ev.index and "Ngay thuong" in ev.index:
        diff = (ev["Cuoi tuan"] - ev["Ngay thuong"]) / ev["Ngay thuong"] * 100
        trend = "cao hon" if diff > 0 else "thap hon"
        print(f"   -> Khung 19h-22h: luu luong cuoi tuan {trend} ngay thuong {abs(diff):.1f}% "
              f"({ev['Cuoi tuan']:.0f} so voi {ev['Ngay thuong']:.0f} xe/gio/camera).")

    print("\n" + "=" * 80)
    print("      HOAN TAT TOAN BO PHAN TICH VA XUAT 6 BIEU DO VAO THU MUC:")
    print(f"      {CHARTS_DIR}")
    print("=" * 80 + "\n")


# ==============================================================================
# 6. HAM DIEU KHIEN CHINH (MAIN)
# ==============================================================================
def main():
    print("#" * 70)
    print("#  PHAN TICH DU LIEU GIAO THONG HA NOI (HANOI TRAFFIC BIG DATA ANALYSIS)")
    print("#" * 70)

    try:
        df_summary, df_violations, df_weather = load_datasets()

        print("\nDang tao 6 bieu do phan tich...")
        chart_01_peak_hours(df_summary)
        chart_02_speed_by_hour(df_summary)
        chart_03_vehicle_types(df_summary)
        chart_04_weather_impact(df_summary)
        chart_05_violations(df_violations)
        chart_06_weekday_weekend(df_summary)

        print_text_summary(df_summary, df_violations, df_weather)

    except Exception as e:
        print(f"\n[LOI]: Co loi xay ra trong qua trinh phan tich: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
