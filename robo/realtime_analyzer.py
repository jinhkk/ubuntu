# realtime_analyzer.py (최종 수정본)

import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
# --- 🔽 한글 폰트 설정을 위한 Pillow 라이브러리 추가 🔽 ---
from PIL import ImageFont, ImageDraw, Image

print("--- 🚀 실시간 속도 분석기 시작 (한글 지원 버전) ---")

try:
    # 1. 저장된 모델 불러오기
    model_filename = "speed_classifier.joblib"
    model = joblib.load(model_filename)
    print(f"✅ 모델 '{model_filename}' 로드 완료!")
except FileNotFoundError:
    print(f"🚨 오류: 모델 파일('{model_filename}')을 찾을 수 없습니다. 'train_model.py'를 먼저 실행해주세요.")
    exit()

# --- 🔽 한글 폰트 경로 지정 (macOS 기본 고딕체) 🔽 ---
# 만약 이 폰트가 없다면, 다른 한글 .ttf 폰트 파일 경로로 지정해주세요.
try:
    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    font = ImageFont.truetype(font_path, 30) # 글씨 크기 30
    status_font = ImageFont.truetype(font_path, 40) # 상태 글씨 크기 40
except FileNotFoundError:
    print(f"🚨 경고: 지정된 폰트 파일을 찾을 수 없습니다. ({font_path})")
    print("한글이 깨질 수 있습니다. 영문으로만 표시합니다.")
    font = None # 폰트가 없으면 None으로 설정

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp_solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("🚨 오류: 웹캠을 열 수 없습니다.")
    exit()

# 실시간 분석을 위한 변수 초기화
prev_landmarks = None
recent_velocities = deque(maxlen=30) 
KEY_JOINTS_TO_TRACK = [
    mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW
]

print("👀 웹캠 분석을 시작합니다. 종료하려면 'q' 키를 누르세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    current_speed_score = 0
    status_kr = "알 수 없음"
    status_en = "Unknown"

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        current_landmarks = results.pose_landmarks.landmark
        
        if prev_landmarks:
            frame_velocity = 0; num_joints = 0
            for joint_index in KEY_JOINTS_TO_TRACK:
                p_curr = current_landmarks[joint_index]; p_prev = prev_landmarks[joint_index]
                distance = np.sqrt((p_curr.x - p_prev.x)**2 + (p_curr.y - p_prev.y)**2)
                frame_velocity += distance; num_joints += 1
            if num_joints > 0: recent_velocities.append(frame_velocity / num_joints)

        prev_landmarks = current_landmarks

        if len(recent_velocities) > 10:
            current_speed_score = np.mean(recent_velocities)
            prediction = model.predict(np.array([[current_speed_score]]))
            if prediction[0] == 1:
                status_kr = "빠름"
                status_en = "Fast"
            else:
                status_kr = "느림"
                status_en = "Slow"

    # --- 🔽 Pillow를 사용한 한글 텍스트 그리기 🔽 ---
    # OpenCV(BGR) 이미지를 Pillow(RGB) 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 텍스트 표시용 검은색 배경 사각형 그리기
    draw.rectangle([(0,0), (400, 100)], fill=(0,0,0))

    # 텍스트 그리기
    score_text = f"Speed Score: {current_speed_score:.5f}"
    color_kr = (0, 255, 0) if status_en == "Slow" else (255, 0, 0) # 느림(초록), 빠름(빨강) - RGB 순서

    if font: # 폰트가 있을 경우 한글로 표시
        draw.text((10, 10), score_text, font=font, fill=(255, 255, 255))
        draw.text((10, 50), f"상태: {status_kr}", font=status_font, fill=color_kr)
    else: # 폰트가 없을 경우 영문으로 표시
        # 이 부분은 cv2.putText를 그대로 사용해도 무방
        cv2.rectangle(frame, (0, 0), (400, 80), (0, 0, 0), -1)
        cv2.putText(frame, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status_en}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if status_en == "Slow" else (0,0,255), 2)
    
    # Pillow(RGB) 이미지를 다시 OpenCV(BGR) 이미지로 변환
    if font:
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    # --- 🔼 Pillow를 사용한 한글 텍스트 그리기 🔼 ---


    cv2.imshow('Real-time Speed Analysis', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("--- 🚀 분석기 종료 ---")