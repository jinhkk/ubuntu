# start_project.launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 로봇 IP 주소를 입력받기 위한 인자(argument) 선언
    declare_robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='172.16.31.8', # 기본값
        description='IP address of the robot'
    )

    # 2. Universal Robot 드라이버 실행
    ur_robot_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ur_robot_driver'), 'launch/ur_control.launch.py')
        ),
        launch_arguments={
            'ur_type': 'ur5e',
            'robot_ip': LaunchConfiguration('robot_ip'), # 입력받은 IP를 드라이버에 전달
            'launch_rviz': 'false'
        }.items()
    )

    # 3. AI 속도 분석 노드 실행 (ai_speed_publisher.py)
    ai_speed_publisher_node = Node(
        package='<your_package_name>', # 나중에 패키지화하면 여기에 패키지 이름을 씁니다.
        executable='ai_speed_publisher.py',
        name='ai_speed_publisher',
        output='screen'
    )

    # 4. UR 로봇 제어 노드 실행 (ur_speed_controller.py)
    ur_speed_controller_node = Node(
        package='<your_package_name>', # 위와 동일
        executable='ur_speed_controller.py',
        name='ur_speed_controller',
        output='screen'
    )

    # 위에서 정의한 모든 액션을 LaunchDescription에 담아 반환
    return LaunchDescription([
        declare_robot_ip_arg,
        ur_robot_driver_launch,
        # 지금은 패키지화 전이므로 아래 두 노드는 별도로 실행합니다.
        # ai_speed_publisher_node, 
        # ur_speed_controller_node
    ])