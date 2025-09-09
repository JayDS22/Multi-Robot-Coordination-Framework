#!/usr/bin/env python3
"""
ROS2 Launch file for Multi-Robot Coordination System
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
import os

def generate_launch_description():
    """Generate launch description for multi-robot coordination system"""
    
    # Declare launch arguments
    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        default_value='5',
        description='Number of robots to launch'
    )
    
    use_simulation_arg = DeclareLaunchArgument(
        'use_simulation',
        default_value='true',
        description='Use simulation mode'
    )
    
    environment_arg = DeclareLaunchArgument(
        'environment',
        default_value='warehouse',
        description='Environment type (warehouse, outdoor, factory)'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='config/system_config.yaml',
        description='Path to system configuration file'
    )
    
    enable_monitoring_arg = DeclareLaunchArgument(
        'enable_monitoring',
        default_value='true',
        description='Enable system monitoring'
    )
    
    enable_visualization_arg = DeclareLaunchArgument(
        'enable_visualization',
        default_value='true',
        description='Enable visualization'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='INFO',
        description='Logging level (DEBUG, INFO, WARNING, ERROR)'
    )
    
    # Get launch configurations
    num_robots = LaunchConfiguration('num_robots')
    use_simulation = LaunchConfiguration('use_simulation')
    environment = LaunchConfiguration('environment')
    config_file = LaunchConfiguration('config_file')
    enable_monitoring = LaunchConfiguration('enable_monitoring')
    enable_visualization = LaunchConfiguration('enable_visualization')
    log_level = LaunchConfiguration('log_level')
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(num_robots_arg)
    ld.add_action(use_simulation_arg)
    ld.add_action(environment_arg)
    ld.add_action(config_file_arg)
    ld.add_action(enable_monitoring_arg)
    ld.add_action(enable_visualization_arg)
    ld.add_action(log_level_arg)
    
    # Launch coordination master
    coordination_master_node = Node(
        package='multi_robot_coordination',
        executable='coordination_master.py',
        name='coordination_master',
        output='screen',
        parameters=[{
            'config_file': config_file,
            'num_robots': num_robots,
            'environment': environment,
            'use_simulation': use_simulation,
            'log_level': log_level
        }],
        arguments=['--config', config_file, '--robots', num_robots, '--environment', environment]
    )
    ld.add_action(coordination_master_node)
    
    # Launch robot agents
    robot_nodes = []
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
                'use_simulation': use_simulation,
                'log_level': log_level
            }],
            arguments=['--robot-id', f'robot_{i}', '--config', 'config/robot_config.yaml'],
            condition=IfCondition(PythonExpression([f'{i}', ' <= ', num_robots]))
        )
        ld.add_action(robot_node)
    
    # Launch task generator
    task_generator_node = Node(
        package='multi_robot_coordination',
        executable='task_generator.py',
        name='task_generator',
        output='screen',
        parameters=[{
            'environment': environment,
            'task_rate': 0.5,
            'complexity': 'medium',
            'use_simulation': use_simulation
        }],
        arguments=['--rate', '0.5', '--complexity', 'medium']
    )
    ld.add_action(task_generator_node)
    
    # Launch system monitor (conditional)
    system_monitor_node = Node(
        package='multi_robot_coordination',
        executable='system_monitor.py',
        name='system_monitor',
        output='screen',
        parameters=[{
            'update_rate': 2.0,
            'enable_visualization': enable_visualization,
            'log_level': log_level
        }],
        condition=IfCondition(enable_monitoring)
    )
    ld.add_action(system_monitor_node)
    
    # Launch RViz for visualization (conditional)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', 'config/multi_robot_visualization.rviz'],
        condition=IfCondition(enable_visualization)
    )
    ld.add_action(rviz_node)
    
    # Launch static transform publisher for coordinate frames
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )
    ld.add_action(static_tf_node)
    
    # Log launch information
    ld.add_action(LogInfo(msg=['Launching Multi-Robot Coordination System']))
    ld.add_action(LogInfo(msg=['Number of robots: ', num_robots]))
    ld.add_action(LogInfo(msg=['Environment: ', environment]))
    ld.add_action(LogInfo(msg=['Simulation mode: ', use_simulation]))
    
    return ld


if __name__ == '__main__':
    generate_launch_description()
