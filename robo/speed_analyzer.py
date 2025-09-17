import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_speed(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    prev_landmarks = None
    velocities = []

    # 추적할 관절 인덱스 (예: 양쪽 손목)
    # MediaPipe Pose Landmaks 참조: https://google.github.io/mediapipe/solutions/pose.html
    # 15: LEFT_WRIST, 16: RIGHT_WRIST
    # 13: LEFT_ELBOW, 14: RIGHT_ELBOW
    KEY_JOINTS_TO_TRACK = [
        mp_pose.PoseLandmark.LEFT_WRIST, 
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 성능 향상을 위해 프레임을 RGB로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            current_landmarks = results.pose_landmarks.landmark
            
            # 이전 프레임의 정보가 있다면 속도 계산
            if prev_landmarks:
                frame_velocity = 0
                num_joints = 0
                
                for joint_index in KEY_JOINTS_TO_TRACK:
                    # 현재 프레임의 관절 좌표
                    p_curr = current_landmarks[joint_index]
                    # 이전 프레임의 관절 좌표
                    p_prev = prev_landmarks[joint_index]

                    # 2D 좌표 거리 계산 (화면상 움직임)
                    distance = np.sqrt(
                        (p_curr.x - p_prev.x)**2 + (p_curr.y - p_prev.y)**2
                    )
                    frame_velocity += distance
                    num_joints += 1
                
                if num_joints > 0:
                    avg_frame_velocity = frame_velocity / num_joints
                    velocities.append(avg_frame_velocity)

            # 현재 랜드마크를 이전 랜드마크로 저장
            prev_landmarks = current_landmarks

            # (디버깅용) 화면에 스켈레톤 그리기
            # frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            # mp_drawing.draw_landmarks(frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # cv2.imshow('Pose Estimation', frame_bgr)

            # if cv2.waitKey(5) & 0xFF == 27: # ESC 누르면 종료
            #     break

    cap.release()
    cv2.destroyAllWindows()

    if not velocities:
        return 0.0

    # 영상의 평균 속도 계산
    average_speed = np.mean(velocities)
    return average_speed

if __name__ == '__main__':
    # 분석할 동영상 파일 경로
    # fast_video.mp4, slow_video.mp4 파일을 준비해야 합니다.
    fast_video_path = 'fast_video.mp4'  # '빠른' 작업 영상 경로
    slow_video_path = 'slow_video.mp4'  # '느린' 작업 영상 경로

    fast_speed_score = calculate_speed(fast_video_path)
    slow_speed_score = calculate_speed(slow_video_path)

    print(f"'{fast_video_path}'의 평균 속도 점수: {fast_speed_score:.5f}")
    print(f"'{slow_video_path}'의 평균 속도 점수: {slow_speed_score:.5f}")
    
    if fast_speed_score is not None and slow_speed_score is not None:
        # 간단한 임계값 설정 예시
        threshold = (fast_speed_score + slow_speed_score) / 2
        print(f"\n설정된 속도 임계값: {threshold:.5f}")
        print("이 값보다 크면 '빠름', 작으면 '느림'으로 분류할 수 있습니다.")