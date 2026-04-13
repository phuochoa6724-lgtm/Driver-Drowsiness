import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np

# --- CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ GIAO DIỆN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "train")  # Thư mục chứa thư mục 'Open_Eyes', 'Closed_Eyes', v.v.
IMG_SIZE = (64, 64)    # Kích thước ảnh (64x64) để mô hình chạy nhẹ và nhanh
BATCH_SIZE = 32
EPOCHS = 50
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "Models", "dms_model_int8.tflite")

def build_model(num_classes=2):
    """
    Xây dựng một mạng CNN siêu nhẹ phù hợp cho Raspberry Pi hoặc Jetson Nano.
    """
    model = models.Sequential([
        # Augmentation nhẹ
        layers.RandomFlip("horizontal", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.RandomBrightness(0.2),
        
        # Lớp Convolution 1
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Lớp Convolution 2
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Lớp Convolution 3
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), # Chống overfitting
        layers.Dense(num_classes, activation='softmax') # Số lượng phân lớp (classes) động
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model

def main():
    print("[INFO] Đang nạp dữ liệu từ thư mục:", DATA_DIR)
    # 1. Tải Dữ Liệu
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"[INFO] Giải mã các nhãn: {class_names}")

    # Tối ưu hóa pipeline dữ liệu để huấn luyện nhanh hơn
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Nén dải màu từ [0,255] về [0,1]
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # 2. Xây dựng và Huấn luyện mô hình
    print("[INFO] Khởi tạo mô hình CNN...")
    model = build_model(len(class_names))
    model.summary()

    print(f"[INFO] Bắt đầu quá trình huấn luyện {EPOCHS} epochs...")
    
    # [TỐI ƯU HÓA] Thêm EarlyStopping để chống Overfitting khi tăng số vòng lặp
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=8, # Sẽ dừng nếu sau 8 epochs mà lỗi validation không giảm
        restore_best_weights=True, # Luôn giữ lại trọng số ở vòng lặp có kết quả tốt nhất
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    print("[INFO] Quá trình huấn luyện hoàn tất.")

    # 3. Chuẩn bị hàm khởi tạo Dataset Đại Điện (Representative Dataset) cho INT8
    # Cần thiết để TensorFLow ánh xạ các giới hạn min/max Float thành Int8 (-128, 127)
    def representative_data_gen():
        for input_value, _ in train_ds.take(100):
            yield [input_value]

    # 4. Chuyển đổi (Export) sang mô hình TFLite được Lượng tử hóa 8-bit (INT8 Quantization)
    print("[INFO] Đang lượng tử hóa (Quantizing) mô hình sáng TFLite 8-bit...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Kích hoạt Optimization DEFAULT để ép kiểu weights và activations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Set representative dataset
    converter.representative_dataset = representative_data_gen
    # Giới hạn I/O và Operation nội suy toàn bộ thành dạng số nguyên (integer) INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # Phù hợp với dải màu ảnh 0-255 lúc truyền vào nếu bạn muốn (thường là uint8)
    converter.inference_output_type = tf.uint8 # Hoặc tf.int8
    
    # Quan trọng: Đảm bảo model hoàn toàn int8, nếu Ops nào ko hỗ trợ sẽ ném lỗi chứ không giữ Float
    converter.allow_custom_ops = False 
    
    # (Tại sao uint8 cho IN/OUT? Thường ảnh cv2 truyền từ 0-255 là mảng np.uint8 nên để uint8 dễ gọi,
    # thay vì int8 -128->127)
    
    tflite_model = converter.convert()

    # Tạo Output Dỉrectory nếu chưa tồn tại
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # 5. Lưu mô hình xuống ổ đĩa
    with open(MODEL_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"[THÀNH CÔNG] 🎉 Đã lưu mạng nhận diện TFLite INT8 tại: {MODEL_SAVE_PATH}")
    print("[HƯỚNG DẪN] Mô hình này sẵn sàng ghép vào luồng xử lý nhận diện Mắt trên Jetson / AI Board.")

if __name__ == '__main__':
    main()
