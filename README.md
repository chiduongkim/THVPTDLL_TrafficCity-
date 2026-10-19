# Hệ Thống Tích Hợp và Phân Tích Dữ Liệu Giao Thông Đô Thị
## Môn: Tích Hợp và Phân Tích Dữ Liệu Lớn

### Mô tả
Hệ thống tích hợp dữ liệu giao thông (camera), thời tiết và vị trí camera,
xử lý bằng công nghệ Big Data (Kafka, Spark), lưu PostgreSQL và trực quan hóa bằng Power BI
để phân tích xu hướng giao thông đô thị tại Hà Nội (5 camera, tháng 9/2024).

### Kiến trúc
```
Data Generator (CSV) ──► Kafka ──► Spark Structured Streaming (demo real-time, in console)
        ▲
Open-Meteo (thời tiết thật)

CSV (data/raw) ──► Spark Batch ──► Parquet + traffic_summary.csv ──► PostgreSQL ──► Power BI
```

### Công nghệ
- **Streaming:** Apache Kafka (3 topic: `traffic-data`, `traffic-violations`, `weather-data`)
- **Big Data Processing:** Apache Spark (PySpark) – batch + structured streaming
- **Database:** PostgreSQL
- **Visualization:** Power BI, matplotlib
- **Container:** Docker Compose (Zookeeper, Kafka, PostgreSQL)

### Nguồn dữ liệu (lưu ý quan trọng)
| Dữ liệu | Bản chất |
|---|---|
| **Thời tiết** (`weather_data.csv`) | **Thật** – Open-Meteo, Hà Nội, 720 giờ tháng 9/2024 |
| **Lượt xe và vi phạm** (`traffic_data.csv`, `violations.csv`) | **Giả lập có tham số** bởi `generate_data.py` (giờ cao điểm, thời tiết, số làn, ngày thường/cuối tuần), khớp từng giờ với thời tiết thật |
| **Vị trí camera** (`cameras.csv`) | 5 nút giao chính tại Hà Nội (tọa độ minh họa) |

> Dự án **chưa** tích hợp camera AI/YOLO thật. Các kết quả phân tích phản ánh giả định của bộ sinh dữ liệu;
> giá trị của dự án nằm ở pipeline tích hợp – xử lý – lưu trữ – trực quan hóa.

### Cách chạy
1. Khởi động hạ tầng: `docker-compose up -d`
2. Cài thư viện: `pip install -r requirements.txt`
3. Đặt file Open-Meteo CSV vào `data/external/open-meteo-21.05N105.81E24m.csv`
   (hoặc truyền `--weather-file <đường_dẫn>`), rồi tạo dữ liệu: `python src/ingestion/generate_data.py`
4. *(Demo streaming)* Terminal 1: `python src/processing/spark_consumer.py`
   → Terminal 2: `python src/ingestion/weather_producer.py` và `python src/ingestion/camera_producer.py`
5. Spark xử lý batch (cần Java; Windows cần `C:\hadoop\bin\winutils.exe`): `python src/processing/spark_batch.py`
   → tạo `data/processed/traffic_summary.csv` và Parquet
6. Nạp vào PostgreSQL: `python src/database/load_to_postgres.py`
   (thay cho ghi JDBC của Spark trên Windows; chạy lại nhiều lần vẫn an toàn)
7. Phân tích và xuất 6 biểu đồ: `python src/analysis/traffic_analysis.py`
   (các truy vấn SQL tương ứng ở `src/analysis/queries.sql`)
8. Mở `dashboards/traffic_dashboard.pbix` và kết nối PostgreSQL (`localhost:5433`, db `traffic_db`)

### Quy ước phân tích
- Mức ùn tắc theo tốc độ trung bình: `<15` critical, `<25` high, `<35` medium, còn lại low.
- Tốc độ trung bình toàn cục/nhóm tính **có trọng số theo số xe** (không dùng trung bình thường của các dòng tổng hợp).
- `weather_type` trong `traffic_summary` giữ nguyên nhãn gốc theo mã WMO của `weather_data` (clear/drizzle/rainy/foggy).

### Hạn chế
- Nhánh streaming mới in ra console, chưa ghi trực tiếp vào PostgreSQL và chưa join với thời tiết.
- Thông tin đăng nhập PostgreSQL (`admin/admin123`) chỉ dùng cho demo cục bộ.

---

## Dự đoán mức ùn tắc sau 1 giờ (benchmark PEMS08)

Mô hình dự đoán **mức ùn tắc của chính cảm biến đó tại t + 1 giờ** từ thông tin có sẵn tại thời điểm t,
so sánh Decision Tree và Random Forest với hai baseline. XGBoost **không dùng** (thư viện không có sẵn trong môi trường dự án).

### Nguồn dữ liệu và ranh giới sử dụng
| | PEMS08 (dùng để huấn luyện/đánh giá) | Dữ liệu Hà Nội của dự án |
|---|---|---|
| Nguồn | Cảm biến đường cao tốc PeMS District 8 | `data/raw/*` |
| Địa điểm | San Bernardino, California, Hoa Kỳ | Hà Nội |
| Thời gian | 1/7/2016 – 31/8/2016, 5 phút/lần (17.856 mốc) | 1–30/9/2024, theo giờ |
| Bản chất giao thông | **Đo thật** (170 cảm biến) | **Mô phỏng** bởi `generate_data.py` |
| Thời tiết | Không có | Open-Meteo Hà Nội (thật) |

- PEMS08 và dữ liệu Hà Nội là **hai bộ riêng**. Cảm biến PEMS không được gán là camera `CAM_HN*`, không có tọa độ hay tên đường Hà Nội, và **không ghép với thời tiết Hà Nội**.
- Mô hình được huấn luyện và đánh giá trên **dữ liệu California**. Đây **chưa phải** mô hình đã kiểm chứng cho Hà Nội.
- File gốc `data/external/pems08/pems08.npz` (kênh theo thứ tự flow, occupancy, speed) lấy từ mã nguồn ASTGCN
  (<https://github.com/Davidham3/ASTGCN-2019-mxnet>, `data/PEMS08/`), không tải trực tiếp từ PeMS; kích thước byte trùng với bản trong các kho STFGNN, STD-MAE, STSGCN_Pytorch. SHA-256 ghi trong `data/processed/pems08/manifest_pems08.json`.
- File NPZ **không chứa timestamp, mã trạm PeMS thật và đơn vị**. Những điều sau là **suy ra**, đã ghi trong manifest:
  - Mốc thời gian bắt đầu 2016-07-01 00:00 (giờ địa phương Bắc Mỹ) theo tài liệu nguồn; kiểm tra bằng dữ liệu: 4/7/2016 (lễ Quốc khánh Mỹ) có lưu lượng thấp nhất, các ngày Chủ nhật thấp nhất còn lại.
  - Tốc độ gốc là **mph** (quy ước PeMS, trung vị 64,9 ≈ giới hạn 65 mph) và được đổi sang km/h (× 1,609344).
  - `sensor_id` = chỉ số cột 0..169 giữ nguyên thứ tự gốc.
- 0,35% ô có flow = 0 và occupancy = 0 (10.560 ô); không phân biệt được "không có xe" với "không đo được". Giữ nguyên, thêm cột `zero_flow_steps` để theo dõi.
- Chưa dùng: thời tiết, sự cố (incident), phân loại xe — PEMS08 không có các thông tin này.

### Nhãn và đặc trưng
- Nhãn (cấu hình `config/congestion.json`, tính theo km/h): **free** > 26; **congested** 16–26 (gồm hai đầu); **severe** < 16.
- Nhãn = lớp của cùng cảm biến sau đúng 1 giờ, dịch chuỗi trong từng sensor: t+4 bước (15 phút) hoặc t+12 bước (5 phút). Các dòng cuối chuỗi không có tương lai bị loại, không backfill.
- Bản 15 phút: flow = **tổng**, occupancy = **trung bình**, speed = **trung bình số học** của 3 mốc 5 phút (không gia quyền theo flow).
- Đặc trưng chỉ dùng thông tin ≤ t của chính cảm biến: tốc độ/flow/occupancy hiện tại, độ trễ (đến 2 giờ), thống kê trượt 1 giờ, giờ trong ngày, thứ trong tuần. Không dùng `sensor_id`.
- Chia theo mốc thời gian 70/15/15 (mọi cảm biến cùng timestamp ở cùng một tập), cách nhau 1 giờ ở cuối train/validation để nhãn không lấy dữ liệu của tập sau. Chọn tham số trên validation; test chỉ đánh giá một lần.

### Cách chạy từ đầu
```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-ml.txt                       # đã kiểm thử với Python 3.14.0
# (một lần) tải dữ liệu vào data/external/pems08/: pems08.npz, distance.csv
python src/ml/build_pems08_dataset.py                    # -> data/processed/pems08/*.parquet + manifest
python src/ml/train_congestion_models.py --resolution 15min   # ~14 phút trên 12 luồng
python src/ml/train_congestion_models.py --resolution 5min    # ~36 phút
python src/ml/bootstrap_ci.py --resolution 15min         # khoảng tin cậy theo khối ngày
python src/ml/audit_congestion_pipeline.py               # kiểm tra độc lập nhãn, rò rỉ, chia tập
docker-compose up -d postgres
python src/database/load_congestion_predictions.py       # nạp dự đoán test vào PostgreSQL
```
Kết nối PostgreSQL đọc từ `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD` (mặc định `localhost:5433`, `traffic_db`, `admin`).
Loader chỉ tạo bảng mới `ml_model_runs` và `ml_congestion_predictions` (migration `src/database/migrations/002_congestion_predictions.sql`),
nạp lại bằng UPSERT theo khóa `(data_source, resolution, model_name, sensor_id, feature_timestamp)` nên chạy lại an toàn; không đụng vào các bảng hiện có.

Đầu ra: `data/processed/pems08/` (dữ liệu, dự đoán test), `models/congestion/pems08_<res>/` (model `.joblib`),
`reports/congestion/pems08_<res>/` (metrics JSON/CSV, confusion matrix, feature importance, tìm kiếm tham số, bootstrap).
Seed = 42; tham số đã chọn và toàn bộ cấu hình ở `metrics.json`.

### Kết quả trên tập test (California, 9–10 ngày cuối 22–31/8/2016)
Phân bố nhãn rất lệch vì ngưỡng 26/16 km/h gần như không bao giờ bị chạm trên đường cao tốc (tốc độ thường 100–110 km/h):

| Tập (15 phút) | free | congested | severe |
|---|---|---|---|
| Train | 705.547 | 452 | 181 |
| Validation | 150.557 | 166 | 67 |
| Test | 151.266 | 112 | 92 |

Bản 5 phút: test có 453.759 / 372 / 279 mẫu.

**15 phút (test, 151.470 mẫu)**

| Mô hình | Accuracy | Balanced acc. | Macro-F1 | F1 congested | F1 severe |
|---|---|---|---|---|---|
| Baseline lớp phổ biến | 0,9987 | 0,333 | 0,333 | 0,000 | 0,000 |
| Baseline giữ nguyên trạng thái hiện tại | 0,9985 | 0,621 | 0,621 | 0,277 | 0,587 |
| Decision Tree | 0,9987 | 0,515 | 0,573 | 0,141 | 0,577 |
| Random Forest | 0,9981 | 0,637 | 0,603 | 0,257 | 0,554 |

**5 phút (test, 454.410 mẫu)**

| Mô hình | Accuracy | Balanced acc. | Macro-F1 | F1 congested | F1 severe |
|---|---|---|---|---|---|
| Baseline lớp phổ biến | 0,9986 | 0,333 | 0,333 | 0,000 | 0,000 |
| Baseline giữ nguyên trạng thái hiện tại | 0,9983 | 0,606 | 0,606 | 0,253 | 0,566 |
| Decision Tree | 0,9987 | 0,449 | 0,510 | 0,095 | 0,436 |
| Random Forest | 0,9978 | 0,640 | 0,602 | 0,251 | 0,557 |

Precision/recall/F1 từng lớp, confusion matrix và số mẫu: `metrics_per_class.csv`, `metrics.json`, `confusion_matrix_test.png`.

### Giới hạn khi diễn giải
- **Accuracy vô nghĩa ở đây** (~99,9% chỉ bằng đoán "free"); dùng macro-F1, balanced accuracy và F1 từng lớp.
- **Random Forest không vượt rõ baseline giữ nguyên trạng thái về macro-F1.** Chênh macro-F1 RF − baseline: −0,018 (15 phút) và −0,004 (5 phút), khoảng tin cậy 95% theo khối ngày đều chứa 0 (`bootstrap_ci_test.json`). RF chỉ hơn baseline ở balanced accuracy (+0,016 [+0,003, +0,043] ở 15 phút; +0,034 [+0,016, +0,074] ở 5 phút; khoảng tin cậy không chứa 0 nhưng chỉ có ~10 khối ngày và mức chênh nhỏ), nhờ recall cao hơn đổi lấy precision thấp hơn; F1 lớp severe không khác biệt rõ. Decision Tree kém baseline. RF tốt hơn Decision Tree (rõ ở bản 5 phút, macro-F1 +0,092 [+0,054, +0,193]).
- **Mẫu dương rất ít và tương quan mạnh:** test chỉ có 204 dòng congested/severe (15 phút). Trên toàn bộ dữ liệu, các dòng ùn thuộc ~290 đợt kéo dài trung bình ~3,7 bước 15 phút, chỉ 62/170 cảm biến từng ùn và 5 cảm biến chiếm 48% số dòng. Khoảng tin cậy theo ngày rất rộng (ví dụ macro-F1 RF 15 phút: 0,603 [0,406, 0,669]).
- Ngưỡng đặt cho đô thị (km/h) áp lên đường cao tốc Mỹ; kết quả không chuyển trực tiếp sang giao thông Hà Nội.
- Random Forest chọn `class_weight=balanced_subsample`: nếu không cân bằng lớp, RF dự đoán gần như toàn "free" (macro-F1 validation ≈ 0,33), nên mô hình đánh đổi precision lấy recall. Decision Tree được chọn không cân bằng lớp (cân bằng làm macro-F1 validation thấp hơn).
- Feature importance (impurity-based) chỉ cho biết mô hình dựa vào biến nào khi dự đoán, **không phải bằng chứng nhân quả**.
- Khoảng cách tới `t+1h` được kiểm tra bằng `audit_congestion_pipeline.py` (tính lại nhãn từ NPZ gốc, thử đổi dữ liệu tương lai không làm đổi đặc trưng).

