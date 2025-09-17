# realtime_analyzer.py

import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque # 최근 데이터를 저장하기 위한 deque 추가

print("--- 🚀 실시간 속도 분석기 시작 ---")

try:
    # 1. 저장된 모델 불러오기
    model_filename = "speed_classifier_augmented.joblib"
    model = joblib.load(model_filename)
    print(f"✅ 모델 '{model_filename}' 로드 완료!")
except FileNotFoundError:
    print(f"🚨 오류: 모델 파일('{model_filename}')을 찾을 수 없습니다. 'train_model.py'를 먼저 실행해주세요.")
    exit()

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 2. 웹캠 열기
cap = cv2.VideoCapture(0) # 0은 기본 웹캠을 의미합니다.
if not cap.isOpened():
    print("🚨 오류: 웹캠을 열 수 없습니다.")
    exit()

# 3. 실시간 분석을 위한 변수 초기화
prev_landmarks = None
# 최근 30 프레임의 속도 데이터를 저장할 공간 (약 1초 분량)
recent_velocities = deque(maxlen=30) 

KEY_JOINTS_TO_TRACK = [
    mp_pose.PoseLandmark.LEFT_WRIST, 
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW
]

print("👀 웹캠 분석을 시작합니다. 종료하려면 'q' 키를 누르세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. 웹캠 영상 처리
    # 거울 모드처럼 보이도록 좌우 반전
    frame = cv2.flip(frame, 1)
    
    # MediaPipe 처리를 위해 이미지 포맷 변경
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