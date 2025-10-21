import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# --- 1. Cấu hình trang (Page Config) ---
# Lệnh này phải là lệnh đầu tiên của Streamlit
st.set_page_config(
    page_title="Nhận diện Món ăn Việt 🍲",
    page_icon="🍲",
    layout="centered"
)

# --- 2. Định nghĩa Tên Lớp & Hằng số ---
# Lấy từ file train.py của bạn: {'Bun': 0, 'Pho': 1, 'comtam': 2}
CLASS_NAMES = ['Bun', 'Pho', 'comtam']
IMG_SIZE = (224, 224)

# --- 3. Tải Model (Phần quan trọng nhất) ---
# @st.cache_resource là cách "Lazy Loading" của Streamlit.
# Nó sẽ chạy hàm này MỘT LẦN DUY NHẤT, tải model,
# và lưu model vào cache (bộ nhớ đệm).
# Lần sau khi app "thức dậy", nó sẽ lấy model từ cache siêu nhanh.
@st.cache_resource
def load_food_model():
    """
    Tải và cache mô hình TensorFlow.
    Hàm này chỉ chạy một lần.
    """
    print("--- ĐANG TẢI MODEL LẦN ĐẦU (sẽ được cache) ---")
    # Đảm bảo đường dẫn này đúng với repo của bạn
    model = tf.keras.models.load_model('model/food_classifier_model.keras')
    print("--- TẢI MODEL THÀNH CÔNG ---")
    return model

# Tải model (sẽ dùng cache nếu đã tải rồi)
model = load_food_model()

# --- 4. Hàm xử lý ảnh (giữ nguyên từ code cũ) ---
def preprocess_image(image_bytes):
    """
    Tiền xử lý ảnh đầu vào để phù hợp với model.
    """
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- 5. Xây dựng Giao diện (UI) ---
st.title("🤖 Nhận diện Phở, Bún, Cơm Tấm")
st.write("Tải ảnh của bạn lên để xem AI dự đoán đây là món gì!")

# 1. Ô tải ảnh
uploaded_file = st.file_uploader("Chọn một ảnh (jpg, jpeg, png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc file ảnh
    image_bytes = uploaded_file.getvalue()
    
    # 2. Hiển thị ảnh
    st.image(image_bytes, caption="Ảnh bạn đã tải lên", use_column_width=True)
    
    # 3. Nút dự đoán
    if st.button("Phân tích ảnh"):
        # Tạo spinner (vòng xoay "Đang phân tích...")
        with st.spinner("Đang phân tích... 🧠"):
            try:
                # Tiền xử lý
                processed_image = preprocess_image(image_bytes)
                
                # Dự đoán
                predictions = model.predict(processed_image)
                
                # Lấy kết quả
                score = np.max(predictions[0])
                class_index = np.argmax(predictions[0])
                class_name = CLASS_NAMES[class_index]
                confidence = float(score) * 100
                
                # 4. Hiển thị kết quả
                st.success(f"Kết quả: **{class_name.upper()}**")
                st.info(f"Độ chính xác: **{confidence:.2f}%**")
                
            except Exception as e:
                st.error(f"Lỗi: Có sự cố xảy ra khi xử lý ảnh. {e}")
