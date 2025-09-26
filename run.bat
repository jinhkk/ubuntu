@echo off
REM 윈도우용 ROS2 환경 설정 스크립트 경로 (설치 위치에 따라 다를 수 있음)
call C:\dev\ros2_humble\local_setup.bat

echo.
echo ========================================
echo  AI Robot Control System
echo ========================================
echo.

REM 사용자에게 로봇 IP 주소 입력받기
set /p ROBOT_IP="Please enter the Robot's IP Address and press Enter: "

echo.
echo Starting the system with Robot IP: %ROBOT_IP%
echo Please start the AI and Controller nodes in separate terminals.
echo.

REM ROS2 런치 파일 실행 (UR 드라이버만)
ros2 launch start_project.launch.py robot_ip:=%ROBOT_IP%

pause