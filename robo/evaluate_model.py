# evaluate_model.py

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import joblib # 모델 로드를 위해 joblib 추가

print("--- 🚀 모델 평가 스크립트 시작 ---")

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# train_model.py와 동일한 calculate_speed 함수
def calculate_speed(video_path):
    video_path_str = str(video_path)
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened(): return None
    velocities = []
    prev_landmarks = None
    KEY_JOINTS_TO_TRACK = [
        mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW
    ]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            current_landmarks = results.pose_landmarks.landmark
            if prev_landmarks:
                frame_velocity = 0; num_joints = 0
                for joint_index in KEY_JOINTS_TO_TRACK:
                    p_curr = current_landmarks[joint_index]; p_prev = prev_landmarks[joint_index]
                    distance = np.sqrt((p_curr.x - p_prev.x)**2 + (p_curr.y - p_prev.y)**2)
                    frame_velocity += distance; num_joints += 1
                if num_joints > 0: velocities.append(frame_velocity / num_joints)
            prev_landmarks = current_landmarks
    cap.release()
    cv2.destroyAllWindows()
    if not velocities: return 0.0
    return np.mean(velocities)

if __name__ == '__main__':
    try:
        # 1. 저장된 모델 불러오기
        model_filename = "speed_classifier.joblib"
        print(f"--- 💾 저장된 모델 '{model_filename}' 로드 중... ---")
        model = joblib.load(model_filename)
        print("✅ 모델 로드 완료!")

        # 2. 평가할 새로운 영상 지정
        # fast 폴더의 1번 영상을 새로운 데이터라고 가정하고 테스트
        script_dir = Path(__file__).resolve().parent
        new_video_path = script_dir / 'fast' / 'fast3.mp4' 
        # (다른 영상을 테스트하고 싶으면 이 경로를 바꾸세요)
        print(f"\n--- 🎞️ 새로운 영상 평가: {new_video_path.name} ---")

        # 3. 새로운 영상에서 특징 추출
        new_score = calculate_speed(new_video_path)
        if new_score is None:
            raise ValueError(f"영상을 처리할 수 없습니다: {new_video_path}")
        
        print(f"계산된 속도 점수: {new_score:.5f}")

        # 4. 불러온 모델로 예측 수행
        # scikit-learn 모델은 2D 배열을 입력으로 기대하므로 형태를 바꿔줌
        input_data = np.array([[new_score]]) 
        prediction = model.predict(input_data)
        result = "빠름" if prediction[0] == 1 else "느림"

        print(f"\n--- ✨ 최종 예측 결과 ---")
        print(f"모델의 예측: '{result}'")

    except FileNotFoundError:
        print(f"\n--- 🚨 오류 발생 🚨 ---")
        print(f"모델 파일('{model_filename}')을 찾을 수 없습니다. 'train_model.py'를 먼저 실행하여 모델을 생성해주세요.")
    except Exception as e:
        print(f"\n--- 🚨 오류 발생 🚨 ---")
        print(f"오류 내용: {e}")