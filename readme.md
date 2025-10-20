# 🤖 Báo cáo Giữa kỳ: Nhận diện Món ăn Việt Nam (Phở, Bún, Cơm Tấm)

Đây là dự án AI cho bài tập giữa kỳ, mục tiêu là xây dựng một mô hình Deep Learning có khả năng phân loại 3 món ăn phổ biến của Việt Nam: Phở, Bún, và Cơm Tấm.

Dự án bao gồm một mô hình đã huấn luyện (sử dụng Transfer Learning với **MobileNetV2**) và một ứng dụng web (sử dụng **Flask**) để demo khả năng dự đoán của mô hình.

---

## Công nghệ sử dụng

* **Ngôn ngữ:** Python 3.12
* **Framework AI:** TensorFlow (cụ thể là `tensorflow-macos` và `tensorflow-metal` để tối ưu cho M1)
* **Backend Web:** Flask
* **Frontend Web:** HTML, CSS, JavaScript
* **Thư viện hỗ trợ:** Pillow, Numpy

---

## 📂 Cấu trúc Thư mục

```
FoodAIP/
├── dataset_3_mon/     # Dữ liệu ảnh thô (đã chia 3-lớp)
├── model/
│   └── food_classifier_model.keras  # Mô hình đã huấn luyện
├── static/            # CSS và JS cho giao diện
│   ├── style.css
│   └── script.js
├── templates/
│   └── index.html     # Giao diện web
├── ven
├── venv               # Môi trường máy ảo venv
├── app.py             # File chạy web server (Flask)
├── train.py           # File huấn luyện mô hình
└── README.md          # File hướng dẫn
```

---

## 🚀 Hướng dẫn Cài đặt và Chạy dự án

Dự án này được phát triển và thử nghiệm trên môi trường **macOS (Apple M1)**.

### 1. Clone Repository

```bash
git clone [URL-repository-cua-ban]
cd [Ten-thu-muc-du-an]
```

### 2. Tạo và Kích hoạt Môi trường Ảo

```bash
# Tạo môi trường ảo (ví dụ tên là 'venv')
python3 -m venv venv

# Kích hoạt môi trường ảo
source venv/bin/activate
```

### 3. Cài đặt Thư viện

Sử dụng file `requirements.txt` để cài đặt tất cả các thư viện cần thiết.

```bash
pip install -r requirements.txt
```

### 4. (Rất quan trọng) Sửa lỗi SSL trên macOS

Nếu đây là lần đầu chạy dự án Python (hoặc dùng Python 3.12+) trên máy, bạn có thể gặp lỗi `[SSL: CERTIFICATE_VERIFY_FAILED]` khi code cố tải mô hình MobileNetV2.

Để khắc phục, vui lòng chạy file sau (chỉ cần chạy 1 lần duy nhất):

1.  Mở **Finder** -> **Applications** -> **Python 3.12** (hoặc phiên bản Python bạn đang dùng).
2.  Nhấp đúp vào file **`Install Certificates.command`**.

---

## 🏃 Cách thức sử dụng

Dự án có 2 chế độ: (1) Chạy web demo với mô hình đã huấn luyện, và (2) Tự huấn luyện lại mô hình.

### 1. Chạy ứng dụng Web (Khuyến nghị)

Mô hình đã được huấn luyện và lưu sẵn trong thư mục `model/food_classifier_model.keras`. Bạn chỉ cần chạy server Flask:

```bash
python app.py
```

Sau khi server khởi động, mở trình duyệt và truy cập:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

Bạn có thể tải ảnh Phở, Bún, Cơm Tấm lên để xem kết quả dự đoán.

### 2. (Tùy chọn) Huấn luyện lại mô hình

Nếu bạn muốn tự huấn luyện lại mô hình từ dữ liệu thô trong `dataset_3_mon`, hãy chạy lệnh sau:

```bash
python train.py
```

Quá trình này sẽ mất vài phút. Sau khi hoàn tất, một file mô hình mới sẽ được tạo và lưu đè vào `model/food_classifier_model.keras`.

---

## 📊 Dataset

* **Nguồn:** Dữ liệu được lấy từ [Kaggle: Vietnamese Foods](https://www.kaggle.com/datasets/quandang/vietnamese-foods).
* **Tiền xử lý:**
    * Chỉ chọn ra 3 lớp: `pho`, `comtam`.
    * Lớp `bun` được gộp chung từ 3 thư mục: `bun bo hue`, `bun cha`, và `bun rieu` để tăng tính đa dạng và đơn giản hóa bài toán.