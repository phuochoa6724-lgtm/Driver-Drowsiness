import numpy as np
from collections import deque
import cv2 # Nhập thư viện xử lý ảnh OpenCV (Computer Vision)
import os

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("[CẢNH BÁO] Không tìm thấy tflite_runtime. Sẽ sử dụng Thuật toán thay thế (Heuristic Fallback) để mô phỏng suy luận AI.")

class DecisionMaker:
    def __init__(self, window_size=60, model_path="Models/dms_model_int8.tflite"):
        """
        Khởi tạo module Ra quyết định với bộ đệm (Buffer) và suy luận AI đa tầng.
        - window_size: số lượng khung hình (frame) lưu trong cửa sổ trượt (trung bình 1-2 giây).
        - model_path: Đường dẫn TFLite.
        """
        self.window_size = window_size
        
        # Khởi tạo các hàng đợi hai đầu (deque) với kích thước cố định.
        self.ear_buffer = deque(maxlen=window_size)
        self.mar_buffer = deque(maxlen=window_size)
        self.pitch_buffer = deque(maxlen=window_size)
        
        # Danh sách các nhãn phân loại đầu ra của hệ thống
        self.labels = ["Normal", "Drowsy", "Yawning", "Talking", "Distracted"]
        
        self.interpreter = None
        # Thiết lập TFLite Interpreter nếu có tflite và file tồn tại
        if TFLITE_AVAILABLE and os.path.exists(model_path):
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as e:
                print(f"[LỖI TFLITE] {e}. Chuyển sang Fallback.")
                self.interpreter = None

    def update_buffer(self, ear, mar, pitch, ear_baseline=0.0, mar_baseline=0.0):
        """
        Cập nhật dữ liệu vào Buffer. Tính toán sự chênh lệch (Delta) so với chuẩn cá nhân.
        """
        self.ear_buffer.append(ear - ear_baseline)
        self.mar_buffer.append(mar - mar_baseline)
        self.pitch_buffer.append(pitch)

    def extract_features(self):
        """
        Trích xuất đặc trưng (Features) từ chuỗi thời gian (time-series).
        """
        # Trả về rỗng nếu chưa thu thập đủ chuỗi thời gian 
        if len(self.ear_buffer) < self.window_size:
            return None 
        
        ear_array = np.array(self.ear_buffer)
        mar_array = np.array(self.mar_buffer)
        pitch_array = np.array(self.pitch_buffer)
        
        ear_mean = np.mean(ear_array)
        mar_mean = np.mean(mar_array)
        
        # Đạo hàm của chuyển động miệng
        mar_grad = np.gradient(mar_array)
        mar_variance = np.var(mar_grad) 
        
        pitch_variance = np.var(pitch_array) 
        
        # Đặc trưng thống kê chuẩn 1D [4 features]
        features = np.array([[ear_mean, mar_mean, mar_variance, pitch_variance]], dtype=np.float32)
        return features

    def _heuristic_fallback(self, features):
        """ Dự đoán nhãn dựa trên quy tắc toán học mềm dẻo (Heuristic) khi không có Model """
        ear_mean, mar_mean, mar_var, pitch_var = features[0]
        
        # Cây quyết định giả lập
        if ear_mean < -0.06:
            return "Drowsy"
        elif mar_mean > 0.25:
            # Ngáp thì miệng mở lớn nhưng biến thiên Gradient nhỏ hơn so với Nói
            return "Yawning"
        elif mar_var > 0.03: 
            # Dao động môi (răng cưa) cực cao
            return "Talking"
        elif pitch_var > 30.0:
            return "Distracted"
            
        return "Normal"

    def predict_state(self):
        """
        Tiến hành chạy mô hình dự đoán (Inference).
        """
        features = self.extract_features()
        if features is None:
            return "Normal"  
        
        state = "Normal"
        # 1. Gọi mạng Neural TFLite nếu có sẵn
        if self.interpreter is not None:
            try:
                self.interpreter.set_tensor(self.input_details[0]['index'], features)
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
                predicted_idx = np.argmax(predictions[0])
                state = self.labels[predicted_idx]
            except Exception as e:
                state = self._heuristic_fallback(features)
        else:
            # 2. Hoặc mô phỏng AI bằng quy tắc tĩnh toán học
            state = self._heuristic_fallback(features)
        
        # 3. Kết hợp Logic (Rule-based) để vá lỗi mô hình nhằm đảm bảo An toàn
        current_pitch = self.pitch_buffer[-1]
        current_ear_delta = self.ear_buffer[-1]
        
        if current_ear_delta < -0.1 and current_pitch > 20.0:  # Mắt sụp VÀ gục đầu
            state = "Drowsy"
            
        return state
