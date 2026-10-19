"""
Spark Batch Processing: Doc CSV, xu ly va ghi vao PostgreSQL.
Phu hop khi da co san du lieu lon.
"""

import os
import sys

# Windows: cau hinh HADOOP_HOME cho PySpark (chi ap dung khi thu muc ton tai)
if os.name == "nt" and os.path.isdir(r"C:\hadoop"):
    os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
    os.environ["PATH"] = os.environ["PATH"] + r";C:\hadoop\bin"

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import (
        col, hour, dayofweek, date_format, count, avg, first,
        when, round as spark_round, to_timestamp, date_trunc, to_date,
        lit, sum as spark_sum
    )
    from pyspark.sql.types import FloatType, IntegerType
except ImportError:
    print("Chua cai pyspark. Chay: pip install pyspark")
    sys.exit(1)

# ============ CAU HINH ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(DATA_PROCESSED, exist_ok=True)

JDBC_URL = "jdbc:postgresql://localhost:5433/traffic_db"
JDBC_PROPERTIES = {
    "user": "admin",
    "password": "admin123",
    "driver": "org.postgresql.Driver"
}


def create_spark():
    """Khoi tao Spark Session."""
    spark = SparkSession.builder \
        .appName("TrafficBigData_HaNoi") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark):
    """Doc du lieu tu CSV."""
    print("\n" + "="*50)
    print("BUOC 1: Doc du lieu")
    print("="*50)

    # Traffic data
    traffic_path = os.path.join(DATA_RAW, "traffic_data.csv")
    df_traffic = spark.read.csv(traffic_path, header=True, inferSchema=True)
    print(f"Traffic data: {df_traffic.count():,} ban ghi")

    # Weather data
    weather_path = os.path.join(DATA_RAW, "weather_data.csv")
    df_weather = spark.read.csv(weather_path, header=True, inferSchema=True)
    print(f"Weather data: {df_weather.count():,} ban ghi")

    # Cameras
    cameras_path = os.path.join(DATA_RAW, "cameras.csv")
    df_cameras = spark.read.csv(cameras_path, header=True, inferSchema=True)
    print(f"Cameras: {df_cameras.count()} camera")

    # Violations
    violations_path = os.path.join(DATA_RAW, "violations.csv")
    df_violations = spark.read.csv(violations_path, header=True, inferSchema=True)
    print(f"Violations: {df_violations.count():,} vi pham")

    return df_traffic, df_weather, df_cameras, df_violations


def process_data(df_traffic, df_weather):
    """Xu ly va tong hop du lieu."""
    print("\n" + "="*50)
    print("BUOC 2: Xu ly du lieu bang Spark")
    print("="*50)

    # Cast timestamp
    df_traffic = df_traffic.withColumn("recorded_at", to_timestamp("timestamp"))
    df_weather = df_weather.withColumn("recorded_at", to_timestamp("timestamp"))

    # Them cot gio, ngay
    df_traffic = df_traffic \
        .withColumn("hour_of_day", hour("recorded_at")) \
        .withColumn("date", date_format("recorded_at", "yyyy-MM-dd")) \
        .withColumn("hour_key", date_trunc("hour", "recorded_at"))

    df_weather = df_weather \
        .withColumn("hour_key", date_trunc("hour", "recorded_at"))

    # Rename weather columns to avoid conflict
    df_weather_renamed = df_weather.select(
        col("hour_key").alias("w_hour_key"),
        col("weather_type").alias("w_weather_type"),
        col("temperature").alias("w_temperature"),
        col("rainfall").alias("w_rainfall")
    )

    # Join traffic voi weather theo hour_key
    df_joined = df_traffic.join(
        df_weather_renamed,
        df_traffic["hour_key"] == df_weather_renamed["w_hour_key"],
        "left"
    )

    print(f"  Da JOIN traffic + weather: {df_joined.count():,} ban ghi")

    # Tong hop theo camera + ngay + gio
    summary = df_joined.groupBy("camera_id", "date", "hour_of_day") \
        .agg(
            count("*").alias("total_vehicles"),
            count(when(col("vehicle_type") == "car", 1)).alias("car_count"),
            count(when(col("vehicle_type") == "motorcycle", 1)).alias("motorcycle_count"),
            count(when(col("vehicle_type") == "bicycle", 1)).alias("bicycle_count"),
            count(when(col("vehicle_type") == "truck", 1)).alias("truck_count"),
            count(when(col("vehicle_type") == "bus", 1)).alias("bus_count"),
            spark_round(avg("speed"), 1).alias("avg_speed"),
            # Vi pham: CSV doc vao Spark thanh NULL khi o trong, nen phai kiem tra
            # ca NULL lan chuoi rong (NULL != "" cho ra NULL, khong dem duoc)
            count(when(col("violation").isNotNull() & (col("violation") != ""), 1)).alias("violation_count"),
            # Thoi tiet: moi gio chi co 1 dong weather nen first() la gia tri duy nhat
            avg("w_temperature").alias("temperature"),
            avg("w_rainfall").alias("rainfall"),
            first("w_weather_type").alias("weather_type"),
        )

    # Giu nguyen weather_type goc (WMO: clear/drizzle/rainy/foggy) tu weather_data
    # thay vi tu suy lai tu luong mua (truoc day lech so voi bang weather_data).
    summary = summary.withColumn(
        "weather_type", when(col("weather_type").isNull(), "clear").otherwise(col("weather_type"))
    )

    # Muc do un tac
    summary = summary.withColumn(
        "congestion_level",
        when(col("avg_speed") < 15, "critical")
        .when(col("avg_speed") < 25, "high")
        .when(col("avg_speed") < 35, "medium")
        .otherwise("low")
    )

    # Rename date -> summary_date va ep kieu DATE de khop schema PostgreSQL
    summary = summary.withColumn("summary_date", to_date(col("date"))).drop("date")
    # Dua summary_date len ngay sau camera_id cho de doc
    summary = summary.select(
        "camera_id", "summary_date", "hour_of_day", "total_vehicles",
        "car_count", "motorcycle_count", "bicycle_count", "truck_count", "bus_count",
        "avg_speed", "violation_count", "temperature", "rainfall",
        "weather_type", "congestion_level"
    )

    print(f"  Da tong hop: {summary.count():,} ban ghi summary")
    print("\n  Mau du lieu summary:")
    summary.show(5, truncate=False)

    return summary


def save_results(spark, summary, df_cameras, df_weather, df_violations):
    """Ghi ket qua vao PostgreSQL va CSV."""
    print("\n" + "="*50)
    print("BUOC 3: Luu ket qua")
    print("="*50)

    # --- Luu CSV backup ---
    summary_csv = os.path.join(DATA_PROCESSED, "traffic_summary.csv")
    summary.toPandas().to_csv(summary_csv, index=False)
    print(f"  Da luu CSV: {summary_csv}")

    # --- Luu Parquet (Data Lake) ---
    parquet_path = os.path.join(DATA_PROCESSED, "traffic_data.parquet")
    try:
        traffic_path = os.path.join(DATA_RAW, "traffic_data.csv")
        df_raw = spark.read.csv(traffic_path, header=True, inferSchema=True)
        df_raw.write.parquet(parquet_path, mode="overwrite")
        print(f"  Da luu Parquet: {parquet_path}")
    except Exception as e:
        print(f"  Loi luu Parquet: {e}")

    # --- Luu vao PostgreSQL ---
    # Dung mode="overwrite" + truncate=true: chi TRUNCATE bang, giu nguyen schema,
    # khoa ngoai va index tu schema.sql (overwrite thuan se DROP/CREATE lai bang,
    # mat khoa ngoai/cot id va loi vi cameras dang duoc bang khac tham chieu).
    # Bang cameras da duoc schema.sql / load_to_postgres.py tao san nen khong ghi de.
    try:
        # Weather
        df_weather_db = df_weather.withColumn("recorded_at", to_timestamp("timestamp"))
        df_weather_db = df_weather_db.withColumn("location", lit("Ha Noi"))
        df_weather_db.select(
            "recorded_at", "temperature", "humidity", "rainfall",
            "weather_type", "wind_speed", "weather_code", "location"
        ).write.option("truncate", "true").jdbc(
            url=JDBC_URL, table="weather_data",
            mode="overwrite", properties=JDBC_PROPERTIES
        )
        print("  Da ghi weather_data vao PostgreSQL")

        # Violations
        df_violations_db = df_violations.withColumn("recorded_at", to_timestamp("timestamp"))
        df_violations_db.select(
            "camera_id", "recorded_at", "vehicle_type",
            "violation_type", "speed"
        ).write.option("truncate", "true").jdbc(
            url=JDBC_URL, table="violations",
            mode="overwrite", properties=JDBC_PROPERTIES
        )
        print(f"  Da ghi {df_violations.count():,} violations vao PostgreSQL")

        # Traffic Summary
        summary.write.option("truncate", "true").jdbc(
            url=JDBC_URL, table="traffic_summary",
            mode="overwrite", properties=JDBC_PROPERTIES
        )
        print(f"  Da ghi {summary.count():,} summary vao PostgreSQL")

        print("\n  THANH CONG: Tat ca du lieu da duoc ghi vao PostgreSQL!")

    except Exception as e:
        print(f"\n  CANH BAO: Khong ket noi duoc PostgreSQL: {e}")
        print("  Du lieu da duoc luu vao CSV/Parquet, co the nap vao DB sau.")
        print("  Dam bao Docker dang chay: docker-compose up -d")


def main():
    print("\n" + "#"*60)
    print("#  SPARK BATCH PROCESSING - GIAO THONG HA NOI")
    print("#"*60)

    spark = create_spark()

    try:
        df_traffic, df_weather, df_cameras, df_violations = load_data(spark)
        summary = process_data(df_traffic, df_weather)
        save_results(spark, summary, df_cameras, df_weather, df_violations)
    finally:
        spark.stop()
        print("\nSpark da dung.")


if __name__ == "__main__":
    main()
