# ai_speed_publisher.py (움직임 없음 감지 기능 추가)

import cv2
import numpy as np
import joblib
from collections import deque
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

class AiSpeedPublisher(Node):

    def __init__(self):
        super().__init__('ai_speed_publisher')
        self.publisher_ = self.create_publisher(String, 'worker_speed', 10)
        timer_period = 0.05
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info("--- AI 속도 분석 노드 (움직임 감지 포함) 초기화 ---")

        try:
            self.model = joblib.load("speed_classifier_simplified.joblib")
            self.scaler = joblib.load("scaler_simplified.joblib")
            self.get_logger().info("모델과 스케일러 로드 완료!")
        except FileNotFoundError:
            self.get_logger().error("모델/스케일러 파일을 찾을 수 없습니다.")
            rclpy.shutdown()
            return

        self.pose = mp_pose.Pose()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다.")
            rclpy.shutdown()
            return

        # '움직임 없음' 상태를 판단하기 위한 임계값
        self.STOP_THRESHOLD = 0.002

        self.prev_landmarks = None
        self.prev_velocity = 0
        self.recent_velocities = deque(maxlen=30)
        self.recent_jerks = deque(maxlen=30)
        self.KEY_JOINTS_TO_TRACK = [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
                                   mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]
        self.get_logger().info("--- 웹캠 분석 시작 ---")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        status_en = "unknown"

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            current_landmarks = results.pose_landmarks.landmark

            if self.prev_landmarks:
                # (속도, 저크 계산 로직은 동일)
                frame_velocity, num_joints = 0, 0
                for joint_index in self.KEY_JOINTS_TO_TRACK:
                    p_curr = current_landmarks[joint_index]; p_prev = self.prev_landmarks[joint_index]
                    distance = np.sqrt((p_curr.x - p_prev.x)**2 + (p_curr.y - p_prev.y)**2)
                    frame_velocity += distance; num_joints += 1
                if num_joints > 0:
                    avg_frame_velocity = frame_velocity / num_joints
                    self.recent_velocities.append(avg_frame_velocity)
                    self.recent_jerks.append(abs(avg_frame_velocity - self.prev_velocity))
                    self.prev_velocity = avg_frame_velocity
            
            self.prev_landmarks = current_landmarks

            if len(self.recent_velocities) > 10:
                speed_score = np.mean(self.recent_velocities)
                
                # ⭐ 새로운 로직: 움직임 없음 판단 ⭐
                if speed_score < self.STOP_THRESHOLD:
                    status_en = "stopped"
                else:
                    # 기존 모델 예측 로직
                    jerk_score = np.mean(self.recent_jerks)
                    input_features = np.array([[speed_score, jerk_score]])
                    input_features_scaled = self.scaler.transform(input_features)
                    prediction = self.model.predict(input_features_scaled)
                    status_en = "fast" if prediction[0] == 1 else "slow"
        else:
             # 사람이 감지되지 않으면 'unknown'
            status_en = "unknown"
            self.prev_landmarks = None


        msg = String()
        msg.data = status_en
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"', throttle_duration_sec=1)

        cv2.putText(frame, f"Status: {status_en}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('AI Speed Publisher Node', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
             rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    ai_speed_publisher = AiSpeedPublisher()
    while rclpy.ok():
        rclpy.spin_once(ai_speed_publisher, timeout_sec=0.01)
    ai_speed_publisher.cap.release()
    cv2.destroyAllWindows()
    ai_speed_publisher.destroy_node()

if __name__ == '__main__':
    main()