#!/usr/bin/env python
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import math
import cv2  # Nhập thư viện xử lý ảnh OpenCV (Computer Vision)
import numpy as np

# Các module Custom
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
from Calibration import Calibrator
from PredictMaker import DecisionMaker

# Khởi tạo bộ phát hiện khuôn mặt của dlib (dựa trên HOG) và sau đó tạo
# bộ dự đoán các điểm mốc trên khuôn mặt (facial landmark predictor)
print("[INFO] Đang nạp bộ dự đoán landmark trên khuôn mặt...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# Khởi tạo mô hình AI và bộ lấy mẫu
print("[INFO] Đang khởi tạo hệ thống AI và Calibrator...")
calibrator = Calibrator(required_frames=100) # Chỉ lấy 100 khung hình làm mẫu ban đầu
decision_maker = DecisionMaker(window_size=30, model_path="Models/dms_model_int8.tflite")

# Khởi tạo luồng video (video stream)
print("[INFO] Khởi động Camera...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

frame_width = 1024
frame_height = 576

# Tọa độ mặc định cho việc vẽ line Đầu (Pitch, Yaw, Roll)
image_points = np.array([
    (359, 391),     # Đỉnh mũi 34
    (399, 561),     # Cằm 9
    (337, 297),     # Góc trái của mắt trái 37
    (513, 301),     # Góc phải của mắt phải 46
    (345, 465),     # Góc trái miệng 49
    (453, 469)      # Góc phải miệng 55
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=1024, height=576)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    rects = detector(gray, 0)

    if len(rects) > 0:
        cv2.putText(frame, f"{len(rects)} face(s) found", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for rect in rects:
        # Tính toán bounding box
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
        
        # Tiên đoán điểm ảnh
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 1. Tính toán Tỷ lệ Mắt (EAR)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 2. Tính toán Tỷ lệ Miệng (MAR)
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # 3. Tính toán góc đầu (Pitch, Yaw, Roll) bằng hàm bổ trợ
        for (i, (x, y)) in enumerate(shape):
            if i == 33: image_points[0] = np.array([x, y], dtype='double')
            elif i == 8: image_points[1] = np.array([x, y], dtype='double')
            elif i == 36: image_points[2] = np.array([x, y], dtype='double')
            elif i == 45: image_points[3] = np.array([x, y], dtype='double')
            elif i == 48: image_points[4] = np.array([x, y], dtype='double')
            elif i == 54: image_points[5] = np.array([x, y], dtype='double')

        (head_tilt_degree, start_point, end_point, end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)
        pitch = head_tilt_degree[0] if head_tilt_degree else 0.0

        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

        # ==========================================
        # AI PIPELINE CHÍNH THỨC BẮT ĐẦU TẠI ĐÂY
        # ==========================================
        if not calibrator.is_calibrated:
            # Nếu chưa có Baseline cá nhân, đưa vào Module hiệu chuẩn (Calibrator)
            calibrator.update(ear, mar)
            progress = calibrator.get_progress() * 100
            cv2.putText(frame, f"CALIBRATING AI... {progress:.1f}%", (320, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        else:
            # Đã có BASELINE. Tính mảng đặc trưng theo Cửa Sổ Trượt (Sliding Window)
            decision_maker.update_buffer(
                ear, mar, pitch, 
                ear_baseline=calibrator.ear_baseline, 
                mar_baseline=calibrator.mar_baseline
            )
            
            # Khởi chạy nội suy AI, lấy nhãn kết quả trạng thái (State)
            state = decision_maker.predict_state()
            
            # Hiển thị độ chênh lệch (Delta) để dev tiện the dõi
            cv2.putText(frame, f"EAR diff: {ear - calibrator.ear_baseline:.3f}", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.putText(frame, f"MAR diff: {mar - calibrator.mar_baseline:.3f}", (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            # Cài đặt mã màu hiển thị thông minh
            color = (0, 255, 0) # Xanh lá (Normal)
            if state in ["Drowsy", "Distracted"]: color = (0, 0, 255) # Đỏ (Nguy hiểm)
            elif state == "Yawning": color = (0, 165, 255) # Cam (Nhắc nhở)
            elif state == "Talking": color = (255, 255, 0) # Xanh Ngọc (Quan sát)
            
            cv2.putText(frame, f"AI STATE: {state}", (350, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Smart AI DMS Architecture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
