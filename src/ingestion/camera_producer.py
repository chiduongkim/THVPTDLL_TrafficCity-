"""
Kafka Producer: doc du lieu tu CSV va gui vao Kafka topics.
Doc file traffic_data.csv va violations.csv, gui tung dong vao Kafka.
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
TRAFFIC_TOPIC = "traffic-data"
VIOLATION_TOPIC = "traffic-violations"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAFFIC_CSV = os.path.join(BASE_DIR, "data", "raw", "traffic_data.csv")


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


def stream_from_csv(producer, filepath, delay=0.001):
    """
    Doc du lieu tu CSV va gui vao Kafka.
    Moi dong = 1 message. Delay de gia lap real-time.
    """
    if not os.path.exists(filepath):
        print(f"Khong tim thay file: {filepath}")
        print("Hay chay generate_data.py truoc!")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        violation_count = 0

        for row in reader:
            # Gui vao topic traffic-data
            message = {
                "camera_id": int(row["camera_id"]),
                "timestamp": row["timestamp"],
                "vehicle_type": row["vehicle_type"],
                "speed": float(row["speed"]),
                "direction": row["direction"]
            }
            producer.send(TRAFFIC_TOPIC, message)

            # Neu co vi pham -> gui vao topic violations
            violation = row.get("violation", "").strip()
            if violation:
                violation_msg = {
                    "camera_id": int(row["camera_id"]),
                    "timestamp": row["timestamp"],
                    "vehicle_type": row["vehicle_type"],
                    "violation_type": violation,
                    "speed": float(row["speed"])
                }
                producer.send(VIOLATION_TOPIC, violation_msg)
                violation_count += 1

            count += 1
            if count % 5000 == 0:
                print(f"  Da gui {count:,} messages ({violation_count:,} vi pham)")

            time.sleep(delay)

    producer.flush()
    print(f"\nHoan tat: {count:,} messages ({violation_count:,} vi pham)")


if __name__ == "__main__":
    print("="*50)
    print("TRAFFIC KAFKA PRODUCER - Ha Noi")
    print("="*50)
    producer = create_producer()
    stream_from_csv(producer, TRAFFIC_CSV)
