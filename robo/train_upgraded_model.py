# train_final_model.py (특징 스케일링 추가)

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# --- StandardScaler 추가 ---
from sklearn.preprocessing import StandardScaler
import joblib

print("--- 최종 모델 학습 스크립트 (특징 스케일링 적용) ---")

# (calculate_features 함수는 이전과 동일합니다)
mp_pose = mp.solutions.pose; pose = mp_pose.Pose()
def calculate_features(video_path):
    video_path_str = str(video_path); cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened(): return None
    velocities, jerks, joint_positions = [], [], []
    prev_landmarks, prev_velocity = None, 0
    KEY_JOINTS_TO_TRACK = [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); results = pose.process(image_rgb)
        if results.pose_landmarks:
            current_landmarks = results.pose_landmarks.landmark
            current_positions = [[pos.x, pos.y] for pos in [current_landmarks[j] for j in KEY_JOINTS_TO_TRACK]]
            joint_positions.append(np.mean(current_positions, axis=0))
            if prev_landmarks:
                frame_velocity = 0; num_joints = 0
                for joint_index in KEY_JOINTS_TO_TRACK:
                    p_curr, p_prev = current_landmarks[joint_index], prev_landmarks[joint_index]
                    distance = np.sqrt((p_curr.x - p_prev.x)**2 + (p_curr.y - p_prev.y)**2)
                    frame_velocity += distance; num_joints += 1
                if num_joints > 0:
                    avg_frame_velocity = frame_velocity / num_joints
                    velocities.append(avg_frame_velocity)
                    jerks.append(abs(avg_frame_velocity - prev_velocity)); prev_velocity = avg_frame_velocity
            prev_landmarks = current_landmarks
    cap.release(); cv2.destroyAllWindows()
    if not velocities: return [0.0, 0.0, 0.0]
    if len(joint_positions) > 1:
        joint_positions = np.array(joint_positions)
        max_x, max_y = np.max(joint_positions, axis=0); min_x, min_y = np.min(joint_positions, axis=0)
        area = (max_x - min_x) * (max_y - min_y)
    else: area = 0.0
    return [np.mean(velocities), np.mean(jerks), area]

if __name__ == '__main__':
    try:
        script_dir = Path(__file__).resolve().parent
        fast_dir, slow_dir = script_dir / 'fast', script_dir / 'slow'
        fast_video_paths = sorted(list(fast_dir.glob('*.mp4')))
        slow_video_paths = sorted(list(slow_dir.glob('*.mp4')))
        if not fast_video_paths or not slow_video_paths: raise ValueError("영상 파일을 찾을 수 없습니다.")

        print("\n--- 다중 특징 추출 진행 ---")
        fast_features = [calculate_features(p) for p in fast_video_paths]
        slow_features = [calculate_features(p) for p in slow_video_paths]
        print("모든 영상의 특징 계산 완료.")

        # 데이터셋 준비
        X = np.array(fast_features + slow_features)
        y = np.array([1] * len(fast_features) + [0] * len(slow_features))

        # --- 특징 스케일링 적용 ---
        print("\n--- ⚖️ 특징 스케일링 적용 ---")
        scaler = StandardScaler()
        # scaler에게 데이터의 분포(평균, 표준편차)를 학습시킴
        scaler.fit(X)
        # 학습된 scaler를 이용해 데이터를 변환 (모든 특징의 단위를 맞춤)
        X_scaled = scaler.transform(X)

        print("\n--- 변환된 특징 데이터 (스케일링 후) ---")
        print("         Speed      Jerk       Area")
        for i, features in enumerate(X_scaled):
            label = "Fast" if y[i] == 1 else "Slow"
            print(f"Video {i+1} ({label}): {features[0]:.5f} | {features[1]:.5f} | {features[2]:.5f}")
        # ---  특징 스케일링 적용 🔼---

        print("\n--- 최종 모델 학습 시작 ---")
        model = LogisticRegression(random_state=42)
        # 스케일링된 데이터를 모델에 학습
        model.fit(X_scaled, y)
        print(" 모델 학습 완료!")

        y_pred = model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        print(f"\n학습 데이터에 대한 예측 정확도: {accuracy * 100:.2f}%")

        # --- 모델과 함께 스케일러도 저장!  ---
        # 실시간 분석 시 똑같은 기준으로 스케일링해야 하므로, 스케일러도 반드시 저장해야 함
        joblib.dump(model, "speed_classifier_final.joblib")
        joblib.dump(scaler, "scaler_final.joblib")
        print(f"\n--- 최종 모델과 스케일러가 저장되었습니다. ---")

    except Exception as e:
        print(f"\n--- 오류 발생 ---")
        print(f"오류 내용: {e}")