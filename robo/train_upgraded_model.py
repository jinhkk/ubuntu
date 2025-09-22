# train_upgraded_model.py

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

print("--- 🚀 업그레이드된 모델 학습 스크립트 시작 (다중 특징) ---")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_features(video_path):
    """
    영상에서 3가지 특징(속도, 저크, 면적)을 계산하는 함수.
    """
    video_path_str = str(video_path)
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened(): return None

    velocities = []
    jerks = []
    joint_positions = []
    
    prev_landmarks = None
    prev_velocity = 0

    KEY_JOINTS_TO_TRACK = [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
                           mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            current_landmarks = results.pose_landmarks.landmark
            
            # 현재 프레임의 관절 위치 저장 (움직임 범위 계산용)
            current_positions = []
            for joint_index in KEY_JOINTS_TO_TRACK:
                pos = current_landmarks[joint_index]
                current_positions.append([pos.x, pos.y])
            joint_positions.append(np.mean(current_positions, axis=0)) # 4개 관절의 중심점 저장

            if prev_landmarks:
                # 1. 속도(Velocity) 계산
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

                    # 2. 저크(Jerk) 계산 (속도의 변화율)
                    jerk = abs(avg_frame_velocity - prev_velocity)
                    jerks.append(jerk)
                    prev_velocity = avg_frame_velocity

            prev_landmarks = current_landmarks

    cap.release()
    cv2.destroyAllWindows()

    if not velocities: return [0.0, 0.0, 0.0]

    # 3. 움직임 범위(Area) 계산
    if len(joint_positions) > 1:
        joint_positions = np.array(joint_positions)
        # 모든 x, y 좌표의 최소/최대값을 찾아 사각형의 너비와 높이를 계산
        max_x, max_y = np.max(joint_positions, axis=0)
        min_x, min_y = np.min(joint_positions, axis=0)
        area = (max_x - min_x) * (max_y - min_y)
    else:
        area = 0.0

    # 3가지 특징의 평균값을 리스트로 반환
    return [np.mean(velocities), np.mean(jerks), area]

if __name__ == '__main__':
    try:
        script_dir = Path(__file__).resolve().parent
        fast_dir = script_dir / 'fast'
        slow_dir = script_dir / 'slow'
        fast_video_paths = sorted(list(fast_dir.glob('*.mp4')))
        slow_video_paths = sorted(list(slow_dir.glob('*.mp4')))
        if not fast_video_paths or not slow_video_paths:
            raise ValueError("영상 파일을 찾을 수 없습니다.")

        print("\n--- 📊 다중 특징 추출 진행 ---")
        fast_features = [calculate_features(p) for p in fast_video_paths]
        slow_features = [calculate_features(p) for p in slow_video_paths]
        print("모든 영상의 특징(속도, 저크, 면적) 계산 완료.")

        # 데이터셋 준비 (이제 X는 3개의 열을 가짐)
        X = np.array(fast_features + slow_features)
        y = np.array([1] * len(fast_features) + [0] * len(slow_features))
        
        # 결과 확인용 출력
        print("\n--- 추출된 특징 데이터 ---")
        print("         Speed      Jerk       Area")
        for i, features in enumerate(X):
            label = "Fast" if y[i] == 1 else "Slow"
            print(f"Video {i+1} ({label}): {features[0]:.5f} | {features[1]:.5f} | {features[2]:.5f}")


        print("\n--- 🤖 업그레이드된 모델 학습 시작 ---")
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        print("✅ 모델 학습 완료!")

        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"\n학습 데이터에 대한 예측 정확도: {accuracy * 100:.2f}%")

        model_filename = "speed_classifier_upgraded.joblib"
        joblib.dump(model, model_filename)
        print(f"\n--- 💾 업그레이드된 모델이 '{model_filename}' 파일로 저장되었습니다. ---")

    except Exception as e:
        print(f"\n--- 🚨 오류 발생 🚨 ---")
        print(f"오류 내용: {e}")