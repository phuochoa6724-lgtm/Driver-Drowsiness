import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# Đường dẫn để lưu model
MODEL_DIR = "Models"
MODEL_PATH = os.path.join(MODEL_DIR, "dms_model_int8.tflite")

def train_dummy_model():
    print("[INFO] Đang tạo mô hình AI mẫu (Dummy Model) để kiểm thử luồng hệ thống...")
    
    # Tạo dữ liệu giả lập (Dummy Data) cho 5 nhãn: Normal, Drowsy, Yawning, Talking, Distracted
    # Mảng Input Feature: [EAR_mean, MAR_mean, MAR_variance, Pitch_variance]
    X_train = np.random.rand(1000, 4).astype(np.float32)
    y_train = np.random.randint(0, 5, size=(1000,))
    
    # Xây dựng mô hình Neural Network thay thế Random Forest để xuất file TFLite dễ dàng
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
                  
    print("[INFO] Đang huấn luyện mô hình (Training)...")
    # Tắt log dài dòng bằng verbose=0
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    # Chuyển đổi mô hình thành TFLite (có áp dụng Quantization INT8 mượt mà)
    print("[INFO] Đang lượng tử hóa (Quantizing INT8) và lưu sang TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Hàm sinh dữ liệu mẫu giúp tối ưu phân bổ trọng số (Post-training quantization)
    def representative_dataset():
        for i in range(100):
            yield [X_train[i:i+1]]
            
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(tflite_model)
        
    print(f"[THÀNH CÔNG] Đã lưu mô hình chuẩn TFLite tại: {MODEL_PATH}")

if __name__ == "__main__":
    train_dummy_model()
