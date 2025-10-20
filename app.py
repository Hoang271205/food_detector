from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# --- 1. Khởi tạo Flask App và Tải Model ---
app = Flask(__name__)
model = tf.keras.models.load_model('model/food_classifier_model.keras')

# --- 2. Định nghĩa Tên Lớp ---
# !!! QUAN TRỌNG !!!
# Thay đổi danh sách này DỰA THEO KẾT QUẢ in ra từ file train.py
# Ví dụ: nếu train.py in ra {'bun': 0, 'com_tam': 1, 'pho': 2}
# thì CLASS_NAMES phải là ['bun', 'com_tam', 'pho']
CLASS_NAMES = ['Bun', 'Pho', 'comtam'] 

# Kích thước ảnh mà model yêu cầu
IMG_SIZE = (224, 224)

# --- 3. Hàm Tiền xử lý ảnh ---
def preprocess_image(image_bytes):
    # Đọc ảnh từ bytes
    img = Image.open(io.BytesIO(image_bytes))
    
    # Chuyển sang RGB (nếu là ảnh RGBA)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Resize ảnh
    img = img.resize(IMG_SIZE)
    
    # Chuyển sang numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Chuẩn hóa (giống lúc train)
    img_array = img_array / 255.0
    
    # Mở rộng chiều (thêm chiều batch)
    # Model yêu cầu input shape là (batch_size, height, width, channels)
    # nên ta đổi (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- 4. Định tuyến (Routing) ---

# Route cho trang chủ (trả về file HTML)
@app.route('/')
def index():
    # Flask sẽ tự động tìm file 'index.html' trong thư mục 'templates'
    return render_template('index.html')

# Route cho API dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Không tìm thấy file ảnh'}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file nào'}), 400

    try:
        # Đọc file ảnh
        image_bytes = file.read()
        
        # Tiền xử lý ảnh
        processed_image = preprocess_image(image_bytes)
        
        # Dự đoán
        predictions = model.predict(processed_image)
        
        # Lấy kết quả
        score = np.max(predictions[0])
        class_index = np.argmax(predictions[0])
        class_name = CLASS_NAMES[class_index]
        
        # Trả về kết quả dạng JSON
        return jsonify({
            'prediction': class_name,
            'confidence': float(score) * 100 # Độ tự tin (percentage)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 5. Chạy App ---
if __name__ == '__main__':
    # Chạy ở chế độ debug để tự động khởi động lại khi có thay đổi
    app.run(debug=True, port=5000)