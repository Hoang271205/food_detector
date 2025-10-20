import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# --- 1. Định nghĩa các hằng số ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = 'dataset_3_mon' # Thư mục chứa 3 thư mục con (bun, com_tam, pho)
EPOCHS = 15 


datagen = ImageDataGenerator(
    rescale=1./255,        
    validation_split=0.2,  
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Vì ta có > 2 lớp
    subset='training'         # Đánh dấu đây là tập training
)

# Tạo bộ dữ liệu kiểm thử (20%)
validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'       # Đánh dấu đây là tập validation
)

# In ra các lớp (classes) mà nó tìm thấy
# QUAN TRỌNG: Ghi nhớ thứ tự này!
# Ví dụ: {'bun': 0, 'com_tam': 1, 'pho': 2}
class_indices = train_generator.class_indices
print("Thứ tự các lớp:", class_indices)
num_classes = len(class_indices)

# --- 3. Xây dựng Mô hình (Transfer Learning) ---

# Tải mô hình MobileNetV2 đã được huấn luyện trước, bỏ đi lớp phân loại cuối cùng
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, # Bỏ lớp fully-connected cuối
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# "Đóng băng" các lớp của base_model, ta không huấn luyện lại chúng
base_model.trainable = False

# Xây dựng phần "đầu" (head) mới cho mô hình
x = base_model.output
x = GlobalAveragePooling2D()(x) # Giảm chiều dữ liệu
x = Dense(128, activation='relu')(x) # Thêm 1 lớp ẩn
x = Dropout(0.5)(x) # Giảm overfitting
# Lớp output: có 'num_classes' (là 3) nơ-ron, dùng 'softmax' cho phân loại đa lớp
predictions = Dense(num_classes, activation='softmax')(x)

# Kết hợp lại thành mô hình cuối cùng
model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. Biên dịch (Compile) Mô hình ---
model.compile(
    optimizer=Adam(learning_rate=0.0001), # Dùng Adam optimizer
    loss='categorical_crossentropy',   # Dùng loss này cho phân loại đa lớp
    metrics=['accuracy']
)

# In cấu trúc mô hình
model.summary()

# --- 5. Huấn luyện Mô hình ---
print("Bắt đầu huấn luyện...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# --- 6. Lưu Mô hình ---
# Tạo thư mục 'model' nếu chưa có
if not os.path.exists('model'):
    os.makedirs('model')

model.save('model/food_classifier_model.keras')
print("Đã huấn luyện và lưu mô hình tại 'model/food_classifier_model.keras'")