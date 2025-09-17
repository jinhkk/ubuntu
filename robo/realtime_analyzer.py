# realtime_analyzer.py (임계값 수동 조정 버전)

import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
from PIL import ImageFont, ImageDraw, Image
import platform

print("--- 🚀 실시간 속도 분석기 시작 (임계값 조정 버전) ---")

try:
    model_filename = "speed_classifier.joblib"
    # 1. 저장된 모델 불러오기
    model_filename = "speed_classifier_augmented.joblib"
    model = joblib.load(model_filename)
    print(f"✅ 모델 '{model_filename}' 로드 완료!")
except FileNotFoundError:
    print(f"🚨 오류: 모델 파일('{model_filename}')을 찾을 수 없습니다.")
    exit()

# --- 🔽 모델의 원래 임계값 계산 🔽 ---
original_threshold = 0
if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
    if model.coef_[0][0] != 0:
        original_threshold = -model.intercept_[0] / model.coef_[0][0]
        print(f"로드된 모델의 원래 임계값: {original_threshold:.5f}")

# --- 🔽 사용자가 원하는 새로운 임계값 설정 🔽 ---
ADJUSTMENT_FACTOR = 1.5 # 이 값을 2.0 (2배), 1.5 (1.5배) 등으로 조절 가능
adjusted_threshold = original_threshold * ADJUSTMENT_FACTOR
print(f"사용자 조정 임계값 ({ADJUSTMENT_FACTOR}배): {adjusted_threshold:.5f}")


# (이하 폰트 설정 및 웹캠 초기화 코드는 이전과 동일)
font_path = None; os_name = platform.system()
if os_name == "Darwin": font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
elif os_name == "Linux": font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
try:
    if font_path:
        font = ImageFont.truetype(font_path, 30); status_font = ImageFont.truetype(font_path, 40)
        print(f"✅ 폰트 로드 완료: {font_path} ({os_name})")
    else: raise OSError
except OSError:
    print(f"🚨 경고: {os_name}에서 한글 폰트를 찾을 수 없습니다. 영문으로만 표시합니다."); font = None
mp_pose = mp.solutions.pose; pose = mp_pose.Pose(); mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(1);
if not cap.isOpened(): print("🚨 오류: 웹캠을 열 수 없습니다."); exit()
prev_landmarks = None; recent_velocities = deque(maxlen=30) 
KEY_JOINTS_TO_TRACK = [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]
print("👀 웹캠 분석을 시작합니다. 종료하려면 'q' 키를 누르세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    current_speed_score = 0
    status = "알 수 없음"

    if results.pose_landmarks:
        # 스켈레톤 그리기
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        current_landmarks = results.pose_landmarks.landmark
        
        if prev_landmarks:
            # 5. 프레임 간 속도 계산
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

        prev_landmarks = current_landmarks

        # 6. 실시간 속도 점수 계산 및 예측
        if len(recent_velocities) > 10: # 최소 10프레임 이상 데이터가 쌓였을 때만 계산
            current_speed_score = np.mean(recent_velocities)
            
            # 모델로 예측
            prediction = model.predict(np.array([[current_speed_score]]))
            status = "fast" if prediction[0] == 1 else "slow"

    # 7. 결과 화면에 표시
    # 상태 표시를 위한 배경 사각형
    cv2.rectangle(frame, (0, 0), (400, 80), (0, 0, 0), -1)
    
    # 속도 점수 텍스트
    score_text = f"Speed Score: {current_speed_score:.5f}"
    cv2.putText(frame, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 상태 텍스트 (빠름/느림)
    status_text = f"Status: {status}"
    color = (0, 255, 0) if status == "slow" else (0, 0, 255) # 느림은 초록색, 빠름은 빨간색
    cv2.putText(frame, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 화면에 영상 출력
    cv2.imshow('Real-time Speed Analysis', frame)

    # 'q' 키를 누르면 루프 탈출
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 8. 자원 해제
cap.release()
cv2.destroyAllWindows()
print("--- 🚀 분석기 종료 ---")