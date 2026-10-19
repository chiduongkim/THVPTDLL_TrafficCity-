"""
Kafka Producer: doc du lieu thoi tiet tu CSV va gui vao Kafka topic.
"""

import json
import csv
import time
import os
import sys

try:
    from kafka import KafkaProducer
except ImportError:
    print("Chua cai kafka-python. Chay: pip install kafka-python")
    sys.exit(1)

# ============ CAU HINH ============
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
WEATHER_TOPIC = "weather-data"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEATHER_CSV = os.path.join(BASE_DIR, "data", "raw", "weather_data.csv")


def create_producer():
    """Tao Kafka Producer."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks='all',
            retries=3
        )
        print(f"Da ket noi Kafka tai {KAFKA_SERVER}")
        return producer
    except Exception as e:
        print(f"Loi ket noi Kafka: {e}")
        sys.exit(1)


def stream_weather(producer, filepath, delay=0.05):
    """
    Doc du lieu thoi tiet tu CSV va gui vao Kafka.
    Moi dong = 1 gio thoi tiet.
    """
    if not os.path.exists(filepath):
        print(f"Khong tim thay file: {filepath}")
        print("Hay chay generate_data.py truoc!")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0

        for row in reader:
            message = {
                "timestamp": row["timestamp"],
                "temperature": float(row["temperature"]),
                "humidity": float(row["humidity"]),
                "rainfall": float(row["rainfall"]),
                "weather_type": row["weather_type"],
                "wind_speed": float(row["wind_speed"]),
                "weather_code": int(row["weather_code"]),
                "location": "Ha Noi"
            }
            producer.send(WEATHER_TOPIC, message)

            count += 1
            if count % 100 == 0:
                print(f"  Da gui {count} ban ghi thoi tiet")

            time.sleep(delay)

    producer.flush()
    print(f"\nHoan tat: {count} ban ghi thoi tiet")


if __name__ == "__main__":
    print("="*50)
    print("WEATHER KAFKA PRODUCER - Ha Noi")
    print("="*50)
    producer = create_producer()
    stream_weather(producer, WEATHER_CSV)
