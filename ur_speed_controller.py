# ur_speed_controller.py (stopped 상태 추가 최종 버전)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class URSpeedController(Node):

    def __init__(self):
        super().__init__('ur_speed_controller')
        self.script_publisher_ = self.create_publisher(String, '/script_command', 10)
        self.subscription = self.create_subscription(
            String,
            'worker_speed',
            self.speed_callback, 
            10)
        self.get_logger().info("--- UR 로봇 속도 제어 노드 (대기 기능 포함) 준비 완료 ---")
        self.last_command_type = None
        self.last_speed_value = None

    def speed_callback(self, msg):
        current_status = msg.data
        
        # 1. 작업자가 움직이는 상태일 때 (fast 또는 slow)
        if current_status in ['fast', 'slow']:
            speed_value = 1.0 if current_status == 'fast' else 0.2
            if self.last_command_type != 'loop' or speed_value != self.last_speed_value:
                self.get_logger().info(f'작업자 움직임 감지. "{current_status}" 속도로 왕복 동작을 시작합니다.')
                ur_script = self.generate_loop_script(speed_value)
                self.send_script(ur_script)
                self.last_command_type = 'loop'
                self.last_speed_value = speed_value

        # ⭐ 2. 작업자가 없거나 움직이지 않을 때 (unknown 또는 stopped) ⭐
        elif current_status in ['unknown', 'stopped']:
            if self.last_command_type != 'wait':
                self.get_logger().info(f'작업자 상태 "{current_status}". 1번 위치로 이동 후 대기합니다.')
                ur_script = self.generate_wait_script()
                self.send_script(ur_script)
                self.last_command_type = 'wait'
    
    def generate_loop_script(self, speed):
        return f"""
def repetitive_move():
  pos1 = p[0.49422, -0.10202, -0.18224, -0.001, 1.624, 0.026]
  pos2 = p[0.21767, -0.38923, -0.16624, 1.179, 1.238, -1.219]
  movel(pos1, a=1.0, v=0.7 * {speed})
  while (True):
    movel(pos2, a=1.0, v=0.7 * {speed})
    movel(pos1, a=1.0, v=0.7 * {speed})
  end
end
repetitive_move()
"""

    def generate_wait_script(self):
        return """
pos1 = p[0.49422, -0.10202, -0.18224, -0.001, 1.624, 0.026]
movel(pos1, a=1.0, v=0.5)
"""

    def send_script(self, script_code):
        msg = String()
        msg.data = script_code
        self.script_publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    ur_speed_controller = URSpeedController()
    rclpy.spin(ur_speed_controller)
    ur_speed_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()