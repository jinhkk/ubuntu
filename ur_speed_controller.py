# ur_speed_controller.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class URSpeedController(Node):

    def __init__(self):
        super().__init__('ur_speed_controller')
        
        # 로봇에게 URScript를 보내기 위한 퍼블리셔
        # 토픽 이름은 반드시 '/script_command' 이어야 합니다.
        self.script_publisher_ = self.create_publisher(String, '/script_command', 10)

        # 'worker_speed' 토픽을 구독하는 서브스크라이버
        self.subscription = self.create_subscription(
            String,
            'worker_speed',
            self.speed_callback, # 메시지를 받으면 speed_callback 함수 실행
            10)
        
        self.get_logger().info("--- UR 로봇 속도 제어 노드 준비 완료 ---")
        self.last_command = None # 마지막으로 보낸 명령을 저장하여 중복 전송 방지
        self.is_robot_moving = False # 로봇이 현재 동작 중인지 상태를 저장

    def speed_callback(self, msg):
        command = msg.data
        
        # 로봇이 움직이는 중이거나, 마지막 명령과 동일하면 새로운 명령을 보내지 않음
        if self.is_robot_moving or command == self.last_command:
            return

        self.get_logger().info(f'작업자 속도 수신: "{command}"')

        # 작업자의 상태에 따라 로봇의 속도를 결정
        # set_speed() 함수는 URScript에서 속도를 변경하는 명령어입니다. (범위: 0~1)
        speed_value = 1.0 if command == 'fast' else 0.3 # 빠를 땐 100%, 느릴 땐 30%
        
        # 로봇이 수행할 동작을 URScript로 정의
        # p[x,y,z,rx,ry,rz]는 좌표와 회전값을 의미합니다.
        # 이 좌표는 실제 로봇의 작업 환경에 맞게 수정해야 합니다.
        ur_script = f"""
def move_box():
  set_speed({speed_value})
  # 1. 상자가 놓이는 위치로 이동
  movel(p[0.1, -0.4, 0.3, 2.2, 2.2, 0], a=0.5, v=0.5)
  # 2. 상자를 밀기 위해 아래로 이동
  movel(p[0.1, -0.4, 0.1, 2.2, 2.2, 0], a=0.5, v=0.5)
  # 3. 앞으로 밀어서 상자를 치움
  movel(p[0.5, -0.4, 0.1, 2.2, 2.2, 0], a=0.5, v=0.5)
  # 4. 원래의 대기 위치로 복귀
  movel(p[0.1, -0.4, 0.3, 2.2, 2.2, 0], a=0.5, v=0.5)
end

move_box()
"""
        
        # URScript를 담은 메시지를 생성하여 퍼블리시
        script_msg = String()
        script_msg.data = ur_script
        self.script_publisher_.publish(script_msg)
        self.get_logger().info(f'{command} 속도({speed_value*100}%)로 동작 스크립트 전송!')
        self.last_command = command
        # 참고: 실제 시스템에서는 로봇의 동작 완료 신호를 받아 is_robot_moving 상태를 제어해야 합니다.
        # 지금은 단순화를 위해 한번 명령을 보내면 다시 보내지 않는 로직만 구현합니다.


def main(args=None):
    rclpy.init(args=args)
    ur_speed_controller = URSpeedController()
    rclpy.spin(ur_speed_controller)
    ur_speed_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()