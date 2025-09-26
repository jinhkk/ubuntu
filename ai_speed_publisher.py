# ai_speed_publisher.py

import cv2
import numpy as np
import joblib
from collections import deque
import platform

import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ROS2 관련 라이브러리 추가
import rclpy
from rclpy.node import Node
from std_msgs.msg import String # 문자열 메시지 타입을 사용

class AiSpeedPublisher(Node):

    def __init__(self):
        # 노드 초기화 및 이름 설정
        super().__init__('ai_speed_publisher')
        
        # 'worker_speed'라는 이름의 토픽으로 String 타입 메시지를 발행하는 퍼블리셔 생성
        self.publisher_ = self.create_publisher(String, 'worker_speed', 10)
        
        # 0.05초 (약 20 FPS) 마다 분석을 수행하도록 타이머 설정
        timer_period = 0.05  
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info("--- AI 속도 분석 노드 초기화 ---")

        # --- 기존 분석 코드 초기화 부분 ---
        try:
            # 논문 및 프로젝트 파일 기반으로 'simplified' 모델을 사용하는 것이 맞습니다.
            self.model = joblib.load("speed_classifier_simplified.joblib")
            self.scaler = joblib.load("scaler_simplified.joblib")
            self.get_logger().info("모델과 스케일러 로드 완료!")
        except FileNotFoundError:
            self.get_logger().error("모델 파일(speed_classifier_simplified.joblib) 또는 스케일러 파일(scaler_simplified.joblib)을 찾을 수 없습니다.")
            self.get_logger().error("스크립트를 실행하는 위치에 해당 파일들이 있는지 확인해주세요.")
            rclpy.shutdown()
            return

        self.pose = mp_pose.Pose()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다.")
            rclpy.shutdown()
            return

        self.prev_landmarks = None
        self.prev_velocity = 0
        self.recent_velocities = deque(maxlen=30)
        self.recent_jerks = deque(maxlen=30)
        self.KEY_JOINTS_TO_TRACK = [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
                                   mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]
        
        self.get_logger().info("--- 웹캠 분석 시작 ---")


    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        status_en = "unknown"

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            current_landmarks = results.pose_landmarks.landmark

            if self.prev_landmarks:
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
                jerk_score = np.mean(self.recent_jerks)
                
                input_features = np.array([[speed_score, jerk_score]])
                input_features_scaled = self.scaler.transform(input_features)
                prediction = self.model.predict(input_features_scaled)

                status_en = "fast" if prediction[0] == 1 else "slow"

        # ROS2 메시지 생성 및 발행
        msg = String()
        msg.data = status_en
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"', throttle_duration_sec=1) # 로그는 1초에 한번만 출력

        # 화면 출력 (디버깅용)
        cv2.putText(frame, f"Status: {status_en}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('AI Speed Publisher Node', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
             rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    ai_speed_publisher = AiSpeedPublisher()
    
    # rclpy.spin()은 노드가 종료될 때까지 대기합니다.
    # 하지만 저희는 OpenCV 창을 닫을 때 노드를 종료시키고 싶으므로, 
    # spin 대신 반복문을 사용합니다.
    while rclpy.ok():
        rclpy.spin_once(ai_speed_publisher, timeout_sec=0.01)

    # 노드 종료 시 자원 해제
    ai_speed_publisher.cap.release()
    cv2.destroyAllWindows()
    ai_speed_publisher.destroy_node()

if __name__ == '__main__':
    main()