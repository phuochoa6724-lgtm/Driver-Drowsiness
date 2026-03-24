import numpy as np

class Calibrator:
    def __init__(self, required_frames=300):
        """
        Khởi tạo hệ thống lấy mẫu (Calibration) để tìm ra thông số sinh trắc học
        tiêu chuẩn (Baseline) của tài xế (mắt híp, răng hô...).
        - required_frames: Số lượng khung hình cần thiết (VD: 300 frame tương đương khoảng 10-15s).
        """
        self.required_frames = required_frames
        self.ear_samples = []
        self.mar_samples = []
        self.is_calibrated = False
        self.ear_baseline = 0.0
        self.mar_baseline = 0.0

    def update(self, ear, mar):
        """
        Nạp dữ liệu vào hệ thống chuẩn hoá trong thời gian đầu.
        Trả về True nếu việc hiệu chuẩn (Calibration) vượt qua ngưỡng yêu cầu và hoàn thành.
        """
        if self.is_calibrated:
            return False
            
        self.ear_samples.append(ear)
        self.mar_samples.append(mar)
        
        if len(self.ear_samples) >= self.required_frames:
            # Lấy trung bình của toàn bộ các khung hình đã thu thập được
            self.ear_baseline = np.mean(self.ear_samples)
            self.mar_baseline = np.mean(self.mar_samples)
            self.is_calibrated = True
            print(f"[CALIBRATION] Đã hiệu chuẩn xong: EAR Cơ sở = {self.ear_baseline:.3f}, MAR Cơ sở = {self.mar_baseline:.3f}")
            return True
            
        return False
        
    def get_progress(self):
        """
        Trả về tiến độ (Tỷ lệ phần trăm 0.0 -> 1.0) của quá trình đo đạc.
        """
        return len(self.ear_samples) / self.required_frames
