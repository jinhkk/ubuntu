# train_simplified_model.py

import cv2
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

print("--- 단순화된 모델 학습 스크립트 (Speed, Jerk 특징) ---")

pose = mp_pose.Pose()

def calculate_features(video_path):
    video_path_str = str(video_path)
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        return None

    velocities = []
    jerks = []
    
    prev_landmarks = None
    prev_velocity = 0

    KEY_JOINTS_TO_TRACK = [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
                           mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            current_landmarks = results.pose_landmarks.landmark
            
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
                    velocities.append(avg_frame_velocity)
                    jerk = abs(avg_frame_velocity - prev_velocity)
                    jerks.append(jerk)
                    prev_velocity = avg_frame_velocity

            prev_landmarks = current_landmarks

    cap.release()
    cv2.destroyAllWindows()

    if not velocities:
        return [0.0, 0.0]

    return [np.mean(velocities), np.mean(jerks)]

if __name__ == '__main__':
    try:
        script_dir = Path(__file__).resolve().parent
        fast_dir = script_dir / 'fast'
        slow_dir = script_dir / 'slow'
        fast_video_paths = sorted(list(fast_dir.glob('*.mp4')))
        slow_video_paths = sorted(list(slow_dir.glob('*.mp4')))
        if not fast_video_paths or not slow_video_paths:
            raise ValueError("영상 파일을 찾을 수 없습니다.")

        print("\n--- 2-Feature 추출 진행 ---")
        fast_features = [calculate_features(p) for p in fast_video_paths]
        slow_features = [calculate_features(p) for p in slow_video_paths]
        print("모든 영상의 특징(Speed, Jerk) 계산 완료.")

        X = np.array(fast_features + slow_features)
        y = np.array([1] * len(fast_features) + [0] * len(slow_features))

        print("\n--- 특징 스케일링 적용 ---")
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        print("\n--- 변환된 특징 데이터 (스케일링 후) ---")
        print("         Speed      Jerk")
        for i, features in enumerate(X_scaled):
            label = "Fast" if y[i] == 1 else "Slow"
            print(f"Video {i+1} ({label}): {features[0]:.5f} | {features[1]:.5f}")

        print("\n--- 단순화된 모델 학습 시작 ---")
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, y)
        print("모델 학습 완료!")

        y_pred = model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        print(f"\n학습 데이터에 대한 예측 정확도: {accuracy * 100:.2f}%")

        joblib.dump(model, "speed_classifier_simplified.joblib")
        joblib.dump(scaler, "scaler_simplified.joblib")
        print(f"\n--- 단순화된 모델과 스케일러가 저장되었습니다. ---")

    except Exception as e:
        print(f"\n--- 오류 발생 ---")
        print(f"오류 내용: {e}")