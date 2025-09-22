# realtime_analyzer_simplified.py

import cv2
import numpy as np
import joblib
from collections import deque
from PIL import ImageFont, ImageDraw, Image
import platform
from pathlib import Path

import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

print("--- 단순화된 실시간 분석기 시작 ---")

try:
    model_filename = "speed_classifier_simplified.joblib"
    scaler_filename = "scaler_simplified.joblib"
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    print(f"모델 '{model_filename}'과 스케일러 '{scaler_filename}' 로드 완료!")
except FileNotFoundError:
    print(f"오류: 모델 또는 스케일러 파일을 찾을 수 없습니다. 'train_simplified_model.py'를 먼저 실행해주세요.")
    exit()

# 폰트 설정
font_path = None
os_name = platform.system()
if os_name == "Darwin":
    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
elif os_name == "Linux":
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

try:
    if font_path and Path(font_path).exists():
        font = ImageFont.truetype(font_path, 24)
        status_font = ImageFont.truetype(font_path, 32)
        print(f"폰트 로드 완료: {font_path}")
    else:
        raise OSError
except (OSError, FileNotFoundError):
    print(f"경고: 한글 폰트를 찾을 수 없습니다. 영문으로만 표시합니다.")
    font = None

pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

# 실시간 분석을 위한 변수들
prev_landmarks = None
prev_velocity = 0
recent_velocities = deque(maxlen=30)
recent_jerks = deque(maxlen=30)
KEY_JOINTS_TO_TRACK = [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
                       mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]

print("웹캠 분석을 시작합니다. 종료하려면 'q' 키를 누르세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    status_kr, status_en = "알 수 없음", "Unknown"
    speed_score, jerk_score = 0.0, 0.0

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        current_landmarks = results.pose_landmarks.landmark

        if prev_landmarks:
            frame_velocity = 0
            num_joints = 0
            for joint_index in KEY_JOINTS_TO_TRACK:
                p_curr, p_prev = current_landmarks[joint_index], prev_landmarks[joint_index]
                distance = np.sqrt((p_curr.x - p_prev.x)**2 + (p_curr.y - p_prev.y)**2)
                frame_velocity += distance
                num_joints += 1
            if num_joints > 0:
                avg_frame_velocity = frame_velocity / num_joints
                recent_velocities.append(avg_frame_velocity)
                recent_jerks.append(abs(avg_frame_velocity - prev_velocity))
                prev_velocity = avg_frame_velocity
        prev_landmarks = current_landmarks

        if len(recent_velocities) > 10:
            speed_score = np.mean(recent_velocities)
            jerk_score = np.mean(recent_jerks)

            input_features = np.array([[speed_score, jerk_score]])
            input_features_scaled = scaler.transform(input_features)
            prediction = model.predict(input_features_scaled)

            if prediction[0] == 1:
                status_kr, status_en = "빠름", "Fast"
            else:
                status_kr, status_en = "느림", "Slow"

    if font:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle([(0, 0), (450, 100)], fill=(0, 0, 0))
        draw.text((10, 5), f"Speed: {speed_score:.4f}", font=font, fill=(255, 255, 255))
        draw.text((10, 35), f"Jerk: {jerk_score:.4f}", font=font, fill=(255, 255, 255))
        color_kr = (0, 255, 0) if status_en == "Slow" else (255, 0, 0)
        draw.text((10, 65), f"상태: {status_kr}", font=status_font, fill=color_kr)
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        cv2.rectangle(frame, (0, 0), (450, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Speed: {speed_score:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Jerk: {jerk_score:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        color_en = (0, 255, 0) if status_en == "Slow" else (0, 0, 255)
        cv2.putText(frame, f"Status: {status_en}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_en, 2)

    cv2.imshow('Simplified Real-time Analyzer', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("--- 분석기 종료 ---")