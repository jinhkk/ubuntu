# realtime_analyzer_upgraded.py (최종 수정본)

import cv2
import numpy as np
import joblib
from collections import deque
from PIL import ImageFont, ImageDraw, Image
import platform

# MediaPipe import 구문 정리
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

print("--- 업그레이드된 실시간 분석기 시작 ---")

try:
    # 이 부분은 train_final_model.py로 생성된 모델과 스케일러를 불러와야 합니다.
    # 파일 이름을 정확히 확인해주세요.
    model_filename = "model1.joblib"
    scaler_filename = "scaler_final.joblib"
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    print(f"모델 '{model_filename}'과 스케일러 '{scaler_filename}' 로드 완료!")
except FileNotFoundError:
    print(f"오류: 모델 또는 스케일러 파일을 찾을 수 없습니다. 'train_final_model.py'를 먼저 실행해주세요.")
    exit()

# 폰트 설정
font_path = None
os_name = platform.system()
if os_name == "Darwin":
    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
elif os_name == "Linux":
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

try:
    if font_path:
        font = ImageFont.truetype(font_path, 24)
        status_font = ImageFont.truetype(font_path, 32)
    else:
        raise OSError
except OSError:
    font = None

# MediaPipe 객체 생성 및 웹캠 초기화
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0) # 사용할 카메라 인덱스
if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

# 실시간 분석을 위한 변수들
prev_landmarks = None
prev_velocity = 0
recent_velocities = deque(maxlen=30)
recent_jerks = deque(maxlen=30)
recent_positions = deque(maxlen=30)

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

    status_kr = "알 수 없음"
    status_en = "Unknown"
    speed_score, jerk_score, area_score = 0.0, 0.0, 0.0

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        current_landmarks = results.pose_landmarks.landmark

        current_positions = []
        for joint_index in KEY_JOINTS_TO_TRACK:
            pos = current_landmarks[joint_index]
            current_positions.append([pos.x, pos.y])
        recent_positions.append(np.mean(current_positions, axis=0))

        if prev_landmarks:
            frame_velocity = 0
            num_joints = 0
            for joint_index in KEY_JOINTS_TO_TRACK:
                p_curr = current_landmarks[joint_index]
                p_prev = prev_landmarks[joint_index]
                distance = np.sqrt((p_curr.x - p_prev.x)**2 + (p_curr.y - p_prev.y)**2)
                frame_velocity += distance
                num_joints += 1
            if num_joints > 0:
                avg_frame_velocity = frame_velocity / num_joints
                recent_velocities.append(avg_frame_velocity)
                jerk = abs(avg_frame_velocity - prev_velocity)
                recent_jerks.append(jerk)
                prev_velocity = avg_frame_velocity

        prev_landmarks = current_landmarks

        if len(recent_velocities) > 10:
            speed_score = np.mean(recent_velocities)
            jerk_score = np.mean(recent_jerks)

            pos_array = np.array(recent_positions)
            max_x, max_y = np.max(pos_array, axis=0)
            min_x, min_y = np.min(pos_array, axis=0)
            area_score = (max_x - min_x) * (max_y - min_y)

            # 실시간 데이터도 학습 때와 동일하게 스케일링
            input_features = np.array([[speed_score, jerk_score, area_score]])
            input_features_scaled = scaler.transform(input_features)
            
            prediction = model.predict(input_features_scaled)

            if prediction[0] == 1:
                status_kr, status_en = "빠름", "Fast"
            else:
                status_kr, status_en = "느림", "Slow"

    # 결과 화면에 표시
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([(0, 0), (450, 130)], fill=(0, 0, 0))

    if font:
        draw.text((10, 5), f"Speed: {speed_score:.4f}", font=font, fill=(255, 255, 255))
        draw.text((10, 35), f"Jerk: {jerk_score:.4f}", font=font, fill=(255, 255, 255))
        draw.text((10, 65), f"Area: {area_score:.4f}", font=font, fill=(255, 255, 255))
        color_kr = (0, 255, 0) if status_en == "Slow" else (255, 0, 0)
        draw.text((10, 95), f"상태: {status_kr}", font=status_font, fill=color_kr)
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow('Upgraded Real-time Analyzer', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("--- 분석기 종료 ---")