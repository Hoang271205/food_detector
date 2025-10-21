import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# --- 1. Cấu hình trang (Page Config) ---
st.set_page_config(
    page_title="Phở-Bún-Cơm AI 🍜",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Dữ liệu & Hằng số ---
CLASS_NAMES = ['Bun', 'Pho', 'comtam']
IMG_SIZE = (224, 224)

# Thêm thông tin về các món ăn để làm app hữu ích hơn
FOOD_INFO = {
    "BUN": {
        "description": "Bún là một món ăn truyền thống với sợi bún gạo, thường được dùng với nước dùng thanh ngọt, thịt, chả, và các loại rau thơm. Có rất nhiều biến thể trên khắp Việt Nam.",
        "calories": "khoảng 400-600 kcal/tô"
    },
    "PHO": {
        "description": "Phở là một trong những món ăn biểu tượng của Việt Nam. Đặc trưng bởi nước dùng đậm đà được ninh từ xương, bánh phở mềm, thịt bò hoặc gà thái mỏng và các loại gia vị.",
        "calories": "khoảng 350-500 kcal/tô"
    },
    "COMTAM": {
        "description": "Cơm tấm, hay cơm tấm Sài Gòn, là món ăn đường phố nổi tiếng. Hạt gạo tấm được nấu chín, ăn kèm với sườn nướng, bì, chả trứng, và nước mắm chua ngọt.",
        "calories": "khoảng 500-700 kcal/đĩa"
    }
}


# --- 3. CSS tùy chỉnh (Phần làm đẹp chính) ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;700&display=swap');

    /* Theme tổng thể */
    body, .stApp {
        font-family: 'Be Vietnam Pro', sans-serif;
    }

    /* Container chính với hiệu ứng "glassmorphism" */
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://images.unsplash.com/photo-1547573882-39002e245aa3?q=80&w=2940&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
    }

    /* Làm mờ sidebar, header và nội dung chính */
    [data-testid="stSidebar"], [data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
    }
    
    .block-container {
        background-color: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    /* Tùy chỉnh nút bấm */
    .stButton > button {
        background-image: linear-gradient(45deg, #FF6B6B, #FFC371);
        color: white;
        border-radius: 50px;
        padding: 12px 30px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(255, 107, 107, 0.75);
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px 0 rgba(255, 107, 107, 0.9);
    }

    /* Tùy chỉnh khu vực tải file */
    [data-testid="stFileUploader"] {
        border: 3px dashed #FF6B6B;
        border-radius: 20px;
        padding: 25px;
        background-color: rgba(255, 255, 255, 0.9);
    }

    /* Hiệu ứng fade-in cho kết quả */
    .result-container {
        animation: fadeIn 1s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# --- 4. Tải Model (Sử dụng Cache) ---
@st.cache_resource
def load_food_model():
    print("--- ĐANG TẢI MODEL LẦN ĐẦU (sẽ được cache) ---")
    model = tf.keras.models.load_model('model/food_classifier_model.keras')
    print("--- TẢI MODEL THÀNH CÔNG ---")
    return model

model = load_food_model()

# --- 5. Hàm xử lý ảnh ---
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- 6. Giao diện (UI) ---

# A. Sidebar
with st.sidebar:
    st.title("🍜 AI Đầu Bếp")
    st.markdown("""
    Chào mừng bạn đến với **AI Đầu Bếp**!
    
    Đây là một mô hình AI được huấn luyện để nhận diện 3 món ăn trứ danh của Việt Nam.
    - **Phở** - Tinh hoa ẩm thực Hà Nội
    - **Bún** - Đa dạng và phong phú
    - **Cơm Tấm** - Hồn cốt ẩm thực Sài Gòn
    
    Hãy tải ảnh của bạn lên và khám phá!
    """)
    st.divider()
# B. Nội dung chính
st.title("📸 Trình Nhận diện Món ăn Việt")
st.markdown("##### *Tải ảnh lên và để AI Đầu Bếp trổ tài!*")

# Container cho phần tải ảnh
uploaded_file = st.file_uploader(
    "Kéo và thả ảnh của bạn vào đây hoặc nhấn để chọn",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    
    col1, col2 = st.columns([0.6, 0.4]) # Cột ảnh lớn hơn một chút
    
    with col1:
        st.image(image_bytes, caption="Ảnh bạn đã tải lên", use_column_width=True)

    with col2:
        if st.button("🍽️ Đây là món gì?"):
            with st.spinner("Đang nấu... à không, đang phân tích... 👨‍🍳"):
                try:
                    processed_image = preprocess_image(image_bytes)
                    predictions = model.predict(processed_image)
                    
                    score = np.max(predictions[0])
                    class_index = np.argmax(predictions[0])
                    class_name = CLASS_NAMES[class_index]
                    confidence = float(score) * 100
                    
                    # Lấy thông tin chi tiết của món ăn
                    food_details = FOOD_INFO.get(class_name.upper(), {
                        "description": "Thông tin chưa được cập nhật.",
                        "calories": "N/A"
                    })
                    
                    # Thêm hiệu ứng fade-in
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.subheader(f"🎉 AI đoán đây là... **{class_name.upper()}**!")
                    
                    # Dùng st.metric để hiển thị đẹp hơn
                    st.metric(label="Độ tự tin", value=f"{confidence:.2f}%")
                    
                    # Thêm thông tin trong một expander
                    with st.expander("📝 Xem thêm thông tin về món này"):
                        st.write(f"**Mô tả:** {food_details['description']}")
                        st.write(f"**Ước tính calo:** {food_details['calories']}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Lỗi: Có sự cố xảy ra. {e}")
else:
    # Hiển thị khi chưa có ảnh
    st.info("💡 Mẹo: Hãy thử với ảnh chụp rõ nét, có đủ ánh sáng để có kết quả tốt nhất!")

