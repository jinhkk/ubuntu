# train_model.py (데이터 증강 기능 추가)

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

print("--- 모델 학습 스크립트 시작 (데이터 증강 포함) ---")


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
def calculate_speed(video_path):
    video_path_str = str(video_path)
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened(): return None
    velocities = []; prev_landmarks = None
    KEY_JOINTS_TO_TRACK = [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]
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
        # 1. 경로 설정 및 파일 로드
        script_dir = Path(__file__).resolve().parent
        fast_dir = script_dir / 'fast'
        slow_dir = script_dir / 'slow'
        fast_video_paths = sorted(list(fast_dir.glob('*.mp4')))
        slow_video_paths = sorted(list(slow_dir.glob('*.mp4')))
        if not fast_video_paths or not slow_video_paths:
            raise ValueError(f"영상 파일을 찾을 수 없습니다.")

        # 2. 특징 추출
        print("\n--- 특징 추출 진행 ---")
        fast_scores = [calculate_speed(p) for p in fast_video_paths]
        slow_scores = [calculate_speed(p) for p in slow_video_paths]
        print("모든 원본 영상의 속도 점수 계산 완료.")

        # --- 데이터 증강 (Data Augmentation)  ---
        print("\n--- 데이터 증강 시작 ---")
        augmentation_factor = 5000 # 각 원본 데이터당 10개의 가상 데이터를 생성
        noise_level = 0.0005 # 노이즈의 강도 (아주 작게 설정)

        augmented_fast_scores = []
        for score in fast_scores:
            noise = np.random.normal(0, noise_level, augmentation_factor)
            augmented_fast_scores.extend(score + noise)

        augmented_slow_scores = []
        for score in slow_scores:
            noise = np.random.normal(0, noise_level, augmentation_factor)
            augmented_slow_scores.extend(score + noise)
        
        total_data_count = len(fast_scores) + len(slow_scores) + len(augmented_fast_scores) + len(augmented_slow_scores)
        print(f"원본 데이터 8개 + 증강된 데이터 {len(augmented_fast_scores) + len(augmented_slow_scores)}개 = 총 {total_data_count}개의 학습 데이터 생성.")
        # --- 데이터 증강 (Data Augmentation) ---

        # 3. 데이터셋 준비 (원본 + 증강 데이터)
        all_fast_scores = fast_scores + augmented_fast_scores
        all_slow_scores = slow_scores + augmented_slow_scores
        
        X = np.array(all_fast_scores + all_slow_scores).reshape(-1, 1)
        y = np.array([1] * len(all_fast_scores) + [0] * len(all_slow_scores))

        # 4. 모델 학습
        print("\n--- LogisticRegression 모델 학습 시작 ---")
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        print("모델 학습 완료!")

        # 5. 모델 평가 및 임계값 확인
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"\n--- 모델 분석 및 평가 ---")
        print(f"증강된 전체 학습 데이터({total_data_count}개)에 대한 예측 정확도: {accuracy * 100:.2f}%")

        if model.coef_[0][0] != 0:
            decision_boundary = -model.intercept_[0] / model.coef_[0][0]
            print(f"모델이 학습한 결정 경계(Threshold): {decision_boundary:.5f}")

        # 6. 모델 파일로 저장
        model_filename = "speed_classifier_augmented.joblib" # 증강 모델은 다른 이름으로 저장
        joblib.dump(model, model_filename)
        print(f"\n--- 모델 저장 완료 ---")
        print(f"증강 데이터로 학습된 모델이 '{model_filename}' 파일로 저장되었습니다.")

    except Exception as e:
        print(f"\n--- 오류 발생 ---")
        print(f"오류 내용: {e}")