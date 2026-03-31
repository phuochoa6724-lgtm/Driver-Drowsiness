import numpy as np
import tensorflow as tf
import os

# 1. Các nhãn trạng thái (Tuân thủ đúng thứ tự trong PredictMaker.py)
labels = ["Normal", "Drowsy", "Yawning", "Talking", "Distracted"]

# 2. Hàm sinh dữ liệu giả lập (Synthesize) dựa vào luật Heuristic hiện tại
# Tính năng: [ear_mean, mar_mean, mar_var, pitch_var, yaw_abs, pitch_raw_mean]
def get_label_by_heuristic(features):
    ear_mean, mar_mean, mar_var, pitch_var, yaw_abs, pitch_raw_mean = features
    
    if ear_mean < -0.06:
        return 1 # Drowsy
    if yaw_abs > 45.0:
        return 4 # Distracted
    if pitch_raw_mean < -40.0 or pitch_raw_mean > 30.0:
        return 4 # Distracted
    if yaw_abs > 30.0 and (pitch_raw_mean < -20.0 or pitch_raw_mean > 15.0):
        return 4 # Distracted
    if mar_mean > 0.25:
        return 2 # Yawning
    if mar_var > 0.03:
        return 3 # Talking
    if pitch_var > 30.0:
        return 4 # Distracted
        
    return 0 # Normal

print("🚀 [1/4] Đang khởi tạo bộ dữ liệu huấn luyện...")
# Sinh 200,000 mẫu ngẫu nhiên cho tập dữ liệu
N_SAMPLES = 200000
X_data = np.zeros((N_SAMPLES, 6), dtype=np.float32)

# Phân bổ ngẫu nhiên dựa trên các khoảng giới hạn thực tế của Camera
X_data[:, 0] = np.random.uniform(-0.15, 0.05, N_SAMPLES) # ear_mean
X_data[:, 1] = np.random.uniform(-0.1, 0.4, N_SAMPLES)   # mar_mean
X_data[:, 2] = np.random.uniform(0.0, 0.1, N_SAMPLES)    # mar_var
X_data[:, 3] = np.random.uniform(0.0, 60.0, N_SAMPLES)   # pitch_var
X_data[:, 4] = np.random.uniform(0.0, 90.0, N_SAMPLES)   # yaw_abs
X_data[:, 5] = np.random.uniform(-60.0, 60.0, N_SAMPLES) # pitch_raw_mean

# Gán nhãn Label y_data
y_data = np.array([get_label_by_heuristic(x) for x in X_data])

# Tách tập train, validation
split_idx = int(0.8 * N_SAMPLES)
X_train, y_train = X_data[:split_idx], y_data[:split_idx]
X_val, y_val = X_data[split_idx:], y_data[split_idx:]

print("🚀 [2/4] Đang thiết kế kiến trúc Mạng nơ-ron đa lớp (Neural Network)...")
# 3. Tạo mô hình Neural Network siêu nhẹ bằng TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax') # Đầu ra 5 nhãn Softmax
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print("🚀 [3/4] Bắt đầu quá trình huấn luyện (Training)...")
# Huấn luyện mô hình (Học từ dữ liệu sinh ra)
history = model.fit(X_train, y_train, epochs=8, batch_size=64, validation_data=(X_val, y_val))

print("🚀 [4/4] Đang đóng gói và giảm dung lượng mô hình ra chuẩn TFLite INT8...")
# 4. Chuyển đổi sang định dạng TFLite Int8
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Cấu hình Quantization xuống Int8 (Chuẩn tối ưu Jetson Nano)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset_gen():
    # Lấy 100 mẫu dữ liệu thực tế để mô phỏng dải Float
    for i in range(100):
        yield [X_train[i].astype(np.float32).reshape(1, 6)]

converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()

# Đảm bảo có thư mục Models/
os.makedirs("Models", exist_ok=True)

# Ghi ra file int8 tflite
model_path = "Models/dms_model_int8.tflite"
with open(model_path, "wb") as f:
    f.write(tflite_quant_model)

print(f"✅ HOÀN TẤT! File mô hình AI đã được xuất tại: {model_path}")
print("   - Kích thước siêu nhẹ, tốc độ inference siêu tốc")
