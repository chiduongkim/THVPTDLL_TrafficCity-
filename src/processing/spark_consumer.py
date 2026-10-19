"""
Spark Structured Streaming: doc tu Kafka,
xu ly va in ket qua ra console.
Dung cho demo real-time.
"""

import os
import sys

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import (
        from_json, col, window, count, avg,
        hour, when, round as spark_round
    )
    from pyspark.sql.types import (
        StructType, StructField, StringType,
        FloatType, IntegerType, TimestampType
    )
except ImportError:
    print("Chua cai pyspark. Chay: pip install pyspark")
    sys.exit(1)

# ============ CAU HINH ============
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")


def main():
    print("\n" + "#"*60)
    print("#  SPARK STREAMING - GIAO THONG HA NOI (REAL-TIME)")
    print("#"*60)

    # Khoi tao Spark
    spark = SparkSession.builder \
        .appName("TrafficStreaming_HaNoi") \
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Schema du lieu traffic
    traffic_schema = StructType([
        StructField("camera_id", IntegerType()),
        StructField("timestamp", StringType()),
        StructField("vehicle_type", StringType()),
        StructField("speed", FloatType()),
        StructField("direction", StringType())
    ])

    # Doc tu Kafka
    print(f"\nDang ket noi Kafka tai {KAFKA_SERVER}...")
    df_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_SERVER) \
        .option("subscribe", "traffic-data") \
        .option("startingOffsets", "latest") \
        .load()

    # Parse JSON
    df_traffic = df_raw \
        .select(from_json(
            col("value").cast("string"), traffic_schema
        ).alias("data")) \
        .select("data.*") \
        .withColumn("event_time", col("timestamp").cast(TimestampType()))

    # Tong hop theo cua so 1 phut
    df_summary = df_traffic \
        .withWatermark("event_time", "2 minutes") \
        .groupBy(
            window("event_time", "1 minute"),
            "camera_id"
        ) \
        .agg(
            count("*").alias("vehicle_count"),
            spark_round(avg("speed"), 1).alias("avg_speed"),
            count(when(col("vehicle_type") == "car", 1)).alias("cars"),
            count(when(col("vehicle_type") == "motorcycle", 1)).alias("motos"),
        )

    # In ra console
    print("Dang lang nghe du lieu tu Kafka...")
    print("(Hay chay camera_producer.py o terminal khac)\n")

    query = df_summary.writeStream \
        .outputMode("update") \
        .format("console") \
        .option("truncate", False) \
        .option("numRows", 20) \
        .trigger(processingTime="10 seconds") \
        .start()

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\nDa dung Spark Streaming.")
        query.stop()
        spark.stop()


if __name__ == "__main__":
    main()
