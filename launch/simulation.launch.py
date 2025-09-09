#!/usr/bin/env python3
"""
ROS2 Launch file for Multi-Robot Coordination Simulation
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    """Generate launch description for simulation"""
    
    # Declare launch arguments
    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        default_value='5',
        description='Number of robots in simulation'
    )
    
    environment_arg = DeclareLaunchArgument(
        'environment',
        default_value='simulation',
        description='Environment type (warehouse, factory, hospital, outdoor, simulation)'
    )
    
    scenario_arg = DeclareLaunchArgument(
        'scenario',
        default_value='basic_navigation',
        description='Test scenario to run'
    )
    
    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Launch GUI components'
    )
    
    visualization_arg = DeclareLaunchArgument(
        'visualization',
        default_value='true',
        description='Enable visualization'
    )
    
    task_rate_arg = DeclareLaunchArgument(
        'task_rate',
        default_value='1.0',
        description='Task generation rate (tasks per second)'
    )
    
    real_time_factor_arg = DeclareLaunchArgument(
        'real_time_factor',
        default_value='1.0',
        description='Simulation speed multiplier'
    )
    
    record_data_arg = DeclareLaunchArgument(
        'record_data',
        default_value='false',
        description='Record simulation data'
    )
    
    # Get launch configurations
    num_robots = LaunchConfiguration('num_robots')
    environment = LaunchConfiguration('environment')
    scenario = LaunchConfiguration('scenario')
    gui = LaunchConfiguration('gui')
    visualization = LaunchConfiguration('visualization')
    task_rate = LaunchConfiguration('task_rate')
    real_time_factor = LaunchConfiguration('real_time_factor')
    record_data = LaunchConfiguration('record_data')
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(coordination_master_node)
    
    # Launch robot agents with delay
    for i in range(1, 11):  # Support up to 10 robots
        robot_node = Node(
            package='multi_robot_coordination',
            executable='robot_agent.py',
            name=f'robot_agent_{i}',
            namespace=f'robot_{i}',
            output='screen',
            parameters=[{
                'robot_id': f'robot_{i}',
                'config_file': 'config/robot_config.yaml',
                'use_simulation': 'true',
                'environment': environment
            }],
            arguments=['--robot-id', f'robot_{i}', '--config', 'config/robot_config.yaml'],
            condition=IfCondition(PythonExpression([f'{i}', ' <= ', num_robots]))
        )
        
        # Add delay between robot launches to prevent startup conflicts
        delayed_robot = TimerAction(
            period=float(i * 0.5),  # 0.5 second delay between robots
            actions=[robot_node]
        )
        ld.add_action(delayed_robot)
    
    # Launch task generator
    task_generator_node = Node(
        package='multi_robot_coordination',
        executable='task_generator.py',
        name='task_generator',
        output='screen',
        parameters=[{
            'environment': environment,
            'scenario': scenario,
            'task_rate': task_rate,
            'use_simulation': 'true'
        }],
        arguments=['--rate', task_rate, '--environment', environment, '--adaptive']
    )
    
    # Delay task generator start until robots are ready
    delayed_task_gen = TimerAction(
        period=5.0,  # Wait 5 seconds for robots to initialize
        actions=[task_generator_node]
    )
    ld.add_action(delayed_task_gen)
    
    # Launch system monitor (conditional on GUI)
    system_monitor_node = Node(
        package='multi_robot_coordination',
        executable='system_monitor.py',
        name='system_monitor',
        output='screen',
        parameters=[{
            'update_rate': 2.0,
            'enable_visualization': visualization,
            'enable_gui': gui,
            'simulation_mode': 'true'
        }],
        condition=IfCondition(gui)
    )
    ld.add_action(system_monitor_node)
    
    # Launch RViz for visualization (conditional)
    rviz_config_file = os.path.join(
        FindPackageShare('multi_robot_coordination').find('multi_robot_coordination'),
        'config',
        'simulation_visualization.rviz'
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(visualization)
    )
    ld.add_action(rviz_node)
    
    # Launch Gazebo simulation (if available)
    gazebo_node = Node(
        package='gazebo_ros',
        executable='gazebo',
        name='gazebo',
        arguments=[
            '--verbose',
            '-s', 'libgazebo_ros_factory.so',
            '-s', 'libgazebo_ros_init.so'
        ],
        output='screen',
        condition=IfCondition(PythonExpression(['"', environment, '" == "simulation"']))
    )
    ld.add_action(gazebo_node)
    
    # Static transform publishers for simulation
    map_to_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )
    ld.add_action(map_to_odom_tf)
    
    # Robot transform publishers
    for i in range(1, 11):
        robot_tf = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'robot_{i}_base_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', f'robot_{i}/base_link'],
            condition=IfCondition(PythonExpression([f'{i}', ' <= ', num_robots]))
        )
        ld.add_action(robot_tf)
    
    # Performance monitor
    performance_monitor_node = Node(
        package='multi_robot_coordination',
        executable='performance_monitor.py',
        name='performance_monitor',
        output='screen',
        parameters=[{
            'monitoring_interval': 5.0,
            'save_results': record_data,
            'simulation_mode': 'true'
        }],
        arguments=['--interval', '5.0']
    )
    ld.add_action(performance_monitor_node)
    
    # Data recorder (conditional)
    data_recorder_node = Node(
        package='rosbag2_py',
        executable='rosbag2',
        name='data_recorder',
        arguments=[
            'record',
            '-a',  # Record all topics
            '-o', f'simulation_data_{environment}_{scenario}',
            '--compression-mode', 'file'
        ],
        condition=IfCondition(record_data)
    )
    ld.add_action(data_recorder_node)
    
    # Simulation controller for automated scenarios
    simulation_controller_node = Node(
        package='multi_robot_coordination',
        executable='simulation_controller.py',
        name='simulation_controller',
        output='screen',
        parameters=[{
            'scenario': scenario,
            'duration': 300.0,  # 5 minutes default
            'auto_shutdown': 'true'
        }],
        arguments=['--scenario', scenario, '--duration', '300']
    )
    ld.add_action(simulation_controller_node)
    
    # Log important information
    ld.add_action(LogInfo(msg=['Launching Multi-Robot Coordination Simulation']))
    ld.add_action(LogInfo(msg=['Environment: ', environment]))
    ld.add_action(LogInfo(msg=['Scenario: ', scenario]))
    ld.add_action(LogInfo(msg=['Number of robots: ', num_robots]))
    ld.add_action(LogInfo(msg=['Task generation rate: ', task_rate, ' tasks/second']))
    ld.add_action(LogInfo(msg=['Real-time factor: ', real_time_factor]))
    
    # Simulation health monitor
    health_monitor_node = Node(
        package='multi_robot_coordination',
        executable='simulation_health_monitor.py',
        name='simulation_health_monitor',
        output='screen',
        parameters=[{
            'check_interval': 10.0,
            'auto_recovery': 'true'
        }]
    )
    ld.add_action(health_monitor_node)
    
    return ld


if __name__ == '__main__':
    generate_launch_description()d_action(num_robots_arg)
    ld.add_action(environment_arg)
    ld.add_action(scenario_arg)
    ld.add_action(gui_arg)
    ld.add_action(visualization_arg)
    ld.add_action(task_rate_arg)
    ld.add_action(real_time_factor_arg)
    ld.add_action(record_data_arg)
    
    # Launch simulation environment
    simulation_node = Node(
        package='multi_robot_coordination',
        executable='simulation_manager.py',
        name='simulation_manager',
        output='screen',
        parameters=[{
            'environment': environment,
            'scenario': scenario,
            'num_robots': num_robots,
            'real_time_factor': real_time_factor,
            'config_file': 'config/environment_config.yaml'
        }],
        arguments=['--environment', environment, '--scenario', scenario]
    )
    ld.add_action(simulation_node)
    
    # Launch coordination master
    coordination_master_node = Node(
        package='multi_robot_coordination',
        executable='coordination_master.py',
        name='coordination_master',
        output='screen',
        parameters=[{
            'config_file': 'config/system_config.yaml',
            'num_robots': num_robots,
            'environment': environment,
            'use_simulation': 'true'
        }],
        arguments=['--config', 'config/system_config.yaml', '--robots', num_robots]
    )
    ld.ad
