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
cap = cv2.VideoCapture(1);   # Mac = 1 웹캠 연결 시 (0) 으로 변경 필
if not cap.isOpened(): print("🚨 오류: 웹캠을 열 수 없습니다."); exit()
prev_landmarks = None; recent_velocities = deque(maxlen=30) 
KEY_JOINTS_TO_TRACK = [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]
print("👀 웹캠 분석을 시작합니다. 종료하려면 'q' 키를 누르세요.")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    current_speed_score = 0
    status_kr = "알 수 없음"; status_en = "Unknown"

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
            
            # --- 🔽 모델 직접 예측 대신, 조정된 임계값으로 판단 🔽 ---
            if current_speed_score > adjusted_threshold:
                status_kr = "빠름"; status_en = "Fast"
            else:
                status_kr = "느림"; status_en = "Slow"
    
    # (이하 텍스트 그리기 및 화면 출력 코드는 이전과 동일)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([(0,0), (500, 100)], fill=(0,0,0))
    score_text = f"Speed Score: {current_speed_score:.5f}"
    threshold_text = f"Threshold: {adjusted_threshold:.5f}" # 현재 적용중인 임계값도 표시
    color_kr = (0, 255, 0) if status_en == "Slow" else (255, 0, 0)
    if font:
        draw.text((10, 5), score_text, font=font, fill=(255, 255, 255))
        draw.text((10, 35), threshold_text, font=font, fill=(255, 255, 150)) # 임계값은 노란색으로
        draw.text((10, 65), f"상태: {status_kr}", font=status_font, fill=color_kr)
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        # (영문 출력 부분은 생략)
        pass
    cv2.imshow('Real-time Speed Analysis', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("--- 🚀 분석기 종료 ---")