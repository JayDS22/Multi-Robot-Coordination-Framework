#!/usr/bin/env python3
"""
Robot Agent for Multi-Robot Coordination Framework
Individual robot agent with reinforcement learning capabilities
"""

import asyncio
import logging
import os
import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml

from algorithms.q_learning import QLearningAgent
from communication.ros_interface import ROSInterface
from communication.fault_tolerance import RobotFaultHandler
from utils.config import ConfigManager
from utils.logger import setup_logger
from utils.metrics import RobotMetrics

@dataclass
class RobotState:
    """Robot state representation"""
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    battery_level: float
    task_load: float
    capabilities: List[str]
    status: str = "idle"  # idle, moving, working, charging
    current_task: Optional[str] = None

class RobotAgent:
    """Individual robot agent with learning capabilities"""
    
    def __init__(self, robot_id: Optional[str] = None, config_file: str = "config/robot_config.yaml"):
        # Set robot ID from environment or parameter
        self.robot_id = robot_id or os.getenv("ROBOT_ID", f"robot_{random.randint(1000, 9999)}")
        
        # Load configuration
        self.config = ConfigManager(config_file)
        self.logger = setup_logger(f"robot_agent_{self.robot_id}")
        
        # Initialize components
        self.q_learning = QLearningAgent(
            agent_id=self.robot_id,
            learning_rate=self.config.get("learning.learning_rate", 0.01),
            discount_factor=self.config.get("learning.discount_factor", 0.95),
            exploration_rate=self.config.get("learning.exploration_rate", 0.15)
        )
        
        self.ros_interface = ROSInterface(node_name=f"robot_{self.robot_id}")
        self.fault_handler = RobotFaultHandler(self.robot_id)
        self.metrics = RobotMetrics(self.robot_id)
        
        # Robot state
        self.state = RobotState(
            position=(0.0, 0.0),
            velocity=(0.0, 0.0),
            battery_level=100.0,
            task_load=0.0,
            capabilities=self.config.get("robot_settings.capabilities", 
                                       ["navigation", "manipulation", "sensing"])
        )
        
        # Robot specifications
        self.max_velocity = self.config.get("robot_settings.max_velocity", 2.0)
        self.sensor_range = self.config.get("robot_settings.sensor_range", 10.0)
        self.communication_range = self.config.get("robot_settings.communication_range", 50.0)
        self.battery_capacity = self.config.get("robot_settings.battery_capacity", 100.0)
        
        # Task management
        self.current_task = None
        self.task_history = []
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        # Performance tracking
        self.start_time = time.time()
        self.total_distance_traveled = 0.0
        self.energy_consumed = 0.0
        self.learning_performance = []
        
        # Communication
        self.master_address = os.getenv("MASTER_IP", "localhost")
        self.heartbeat_interval = 1.0
        self.last_heartbeat = time.time()
        
        self.logger.info(f"Robot agent {self.robot_id} initialized with capabilities: {self.state.capabilities}")
    
    async def start(self):
        """Start the robot agent"""
        self.logger.info(f"Starting robot agent {self.robot_id}")
        
        # Initialize ROS interface
        await self.ros_interface.initialize()
        
        # Register with coordination master
        await self.register_with_master()
        
        # Start concurrent tasks
        tasks = [
            self.heartbeat_sender(),
            self.task_listener(),
            self.state_updater(),
            self.learning_updater(),
            self.fault_monitor(),
            self.battery_monitor()
        ]
        
        await asyncio.gather(*tasks)
    
    async def register_with_master(self):
        """Register with the coordination master"""
        try:
            registration_data = {
                'robot_id': self.robot_id,
                'capabilities': self.state.capabilities,
                'position': self.state.position,
                'specifications': {
                    'max_velocity': self.max_velocity,
                    'sensor_range': self.sensor_range,
                    'communication_range': self.communication_range,
                    'battery_capacity': self.battery_capacity
                }
            }
            
            success = await self.ros_interface.register_with_master(registration_data)
            
            if success:
                self.logger.info("Successfully registered with coordination master")
            else:
                self.logger.error("Failed to register with coordination master")
                
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
    
    async def heartbeat_sender(self):
        """Send periodic heartbeat to coordination master"""
        while True:
            try:
                heartbeat_data = {
                    'robot_id': self.robot_id,
                    'timestamp': time.time(),
                    'position': self.state.position,
                    'battery_level': self.state.battery_level,
                    'status': self.state.status,
                    'task_load': self.state.task_load,
                    'current_task': self.current_task.task_id if self.current_task else None
                }
                
                await self.ros_interface.send_heartbeat(heartbeat_data)
                self.last_heartbeat = time.time()
                
            except Exception as e:
                self.logger.warning(f"Heartbeat failed: {e}")
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def task_listener(self):
        """Listen for task assignments from coordination master"""
        while True:
            try:
                task = await self.ros_interface.receive_task_assignment()
                
                if task:
                    await self.handle_task_assignment(task)
                    
            except Exception as e:
                self.logger.error(f"Task listener error: {e}")
            
            await asyncio.sleep(0.1)
    
    async def handle_task_assignment(self, task):
        """Handle incoming task assignment"""
        self.logger.info(f"Received task assignment: {task['task_id']} - {task['task_type']}")
        
        # Check if robot can handle the task
        if not self.can_execute_task(task):
            self.logger.warning(f"Cannot execute task {task['task_id']} - insufficient capabilities")
            await self.reject_task(task)
            return
        
        # Accept task
        self.current_task = task
        self.state.status = "assigned"
        
        # Send acceptance confirmation
        await self.ros_interface.confirm_task_assignment(task['task_id'])
        
        # Start task execution
        await self.execute_task(task)
    
    def can_execute_task(self, task) -> bool:
        """Check if robot can execute the given task"""
        # Check capabilities
        required_capabilities = task.get('required_capabilities', [])
        if not all(cap in self.state.capabilities for cap in required_capabilities):
            return False
        
        # Check battery level
        estimated_energy = self.estimate_task_energy(task)
        if self.state.battery_level < estimated_energy * 1.2:  # 20% safety margin
            return False
        
        # Check distance feasibility
        distance_to_task = self.calculate_distance(self.state.position, task['location'])
        if distance_to_task > self.communication_range:
            return False
        
        return True
    
    def estimate_task_energy(self, task) -> float:
        """Estimate energy required for task execution"""
        distance = self.calculate_distance(self.state.position, task['location'])
        
        # Energy for movement (simplified model)
        movement_energy = distance * 0.1  # 0.1% per meter
        
        # Energy for task execution
        task_energy = task.get('estimated_duration', 10.0) * 0.05  # 0.05% per second
        
        return movement_energy + task_energy
    
    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    async def execute_task(self, task):
        """Execute assigned task"""
        try:
            self.logger.info(f"Starting execution of task {task['task_id']}")
            self.state.status = "working"
            task_start_time = time.time()
            
            # Move to task location
            await self.move_to_location(task['location'])
            
            # Perform task-specific actions
            success = await self.perform_task_actions(task)
            
            # Calculate execution time and energy consumption
            execution_time = time.time() - task_start_time
            energy_used = self.estimate_task_energy(task)
            self.state.battery_level = max(0, self.state.battery_level - energy_used)
            self.energy_consumed += energy_used
            
            # Update learning algorithm
            reward = self.calculate_task_reward(task, success, execution_time)
            self.q_learning.update_q_value(task['task_type'], 'execute', reward)
            
            # Report completion
            await self.report_task_completion(task, success)
            
            # Update metrics
            if success:
                self.completed_tasks += 1
                self.logger.info(f"Task {task['task_id']} completed successfully")
            else:
                self.failed_tasks += 1
                self.logger.warning(f"Task {task['task_id']} failed")
            
            # Reset state
            self.current_task = None
            self.state.status = "idle"
            self.state.task_load = 0.0
            
            # Record task in history
            self.task_history.append({
                'task_id': task['task_id'],
                'task_type': task['task_type'],
                'success': success,
                'execution_time': execution_time,
                'energy_used': energy_used,
                'reward': reward,
                'timestamp': time.time()
            })
            
        except Exception as e:
            self.logger.error(f"Task execution error: {e}")
            await self.report_task_completion(task, False)
            self.current_task = None
            self.state.status = "idle"
            self.failed_tasks += 1
    
    async def move_to_location(self, target_location: Tuple[float, float]):
        """Move robot to target location"""
        start_pos = self.state.position
        distance = self.calculate_distance(start_pos, target_location)
        
        if distance < 0.1:  # Already at location
            return
        
        self.logger.debug(f"Moving to location {target_location}, distance: {distance:.2f}m")
        self.state.status = "moving"
        
        # Simulate movement (in real implementation, this would control actual robot)
        travel_time = distance / self.max_velocity
        steps = max(10, int(travel_time * 10))  # 10 steps per second
        
        for i in range(steps):
            # Linear interpolation for smooth movement
            progress = (i + 1) / steps
            new_x = start_pos[0] + (target_location[0] - start_pos[0]) * progress
            new_y = start_pos[1] + (target_location[1] - start_pos[1]) * progress
            
            self.state.position = (new_x, new_y)
            
            # Update velocity
            if i < steps - 1:
                next_progress = (i + 2) / steps
                next_x = start_pos[0] + (target_location[0] - start_pos[0]) * next_progress
                next_y = start_pos[1] + (target_location[1] - start_pos[1]) * next_progress
                
                self.state.velocity = (
                    (next_x - new_x) * 10,  # velocity in m/s
                    (next_y - new_y) * 10
                )
            else:
                self.state.velocity = (0.0, 0.0)
            
            await asyncio.sleep(travel_time / steps)
        
        self.total_distance_traveled += distance
        self.logger.debug(f"Arrived at location {target_location}")
    
    async def perform_task_actions(self, task) -> bool:
        """Perform task-specific actions"""
        task_type = task['task_type']
        duration = task.get('estimated_duration', 10.0)
        
        self.logger.debug(f"Performing {task_type} for {duration}s")
        
        # Simulate task execution with learning-based performance
        base_success_rate = 0.85
        learning_bonus = self.q_learning.get_q_value(task_type, 'execute') * 0.1
        success_rate = min(0.95, base_success_rate + learning_bonus)
        
        # Simulate task execution time
        for i in range(int(duration)):
            self.state.task_load = (i + 1) / duration
            await asyncio.sleep(1.0)
            
            # Check for interruptions (battery, faults)
            if self.state.battery_level < 10.0:
                self.logger.warning("Task interrupted due to low battery")
                return False
        
        # Determine success based on learning and random factors
        success = random.random() < success_rate
        
        return success
    
    def calculate_task_reward(self, task, success: bool, execution_time: float) -> float:
        """Calculate reward for task execution"""
        base_reward = 1.0 if success else -0.5
        
        # Time efficiency bonus/penalty
        expected_time = task.get('estimated_duration', 10.0)
        time_efficiency = expected_time / max(execution_time, 0.1)
        time_bonus = (time_efficiency - 1.0) * 0.3
        
        # Priority bonus
        priority_bonus = task.get('priority', 1.0) * 0.2
        
        # Energy efficiency bonus
        energy_efficiency = 1.0 - (self.energy_consumed / self.battery_capacity)
        energy_bonus = energy_efficiency * 0.1
        
        total_reward = base_reward + time_bonus + priority_bonus + energy_bonus
        return max(-1.0, min(2.0, total_reward))  # Clamp between -1 and 2
    
    async def report_task_completion(self, task, success: bool):
        """Report task completion to coordination master"""
        completion_data = {
            'robot_id': self.robot_id,
            'task_id': task['task_id'],
            'success': success,
            'completion_time': time.time(),
            'energy_used': self.estimate_task_energy(task),
            'execution_details': {
                'final_position': self.state.position,
                'battery_level': self.state.battery_level
            }
        }
        
        await self.ros_interface.report_task_completion(completion_data)
    
    async def reject_task(self, task):
        """Reject task assignment"""
        rejection_data = {
            'robot_id': self.robot_id,
            'task_id': task['task_id'],
            'reason': 'insufficient_capabilities_or_resources'
        }
        
        await self.ros_interface.reject_task_assignment(rejection_data)
    
    async def state_updater(self):
        """Update robot state periodically"""
        while True:
            await asyncio.sleep(0.5)
            
            # Update battery drain based on activity
            battery_drain_rate = self.get_battery_drain_rate()
            self.state.battery_level = max(0, self.state.battery_level - battery_drain_rate * 0.5)
            
            # Update position noise (sensor uncertainty)
            if self.state.status == "idle":
                # Add small amount of noise to simulate sensor drift
                noise_x = random.gauss(0, 0.01)
                noise_y = random.gauss(0, 0.01)
                self.state.position = (
                    self.state.position[0] + noise_x,
                    self.state.position[1] + noise_y
                )
    
    def get_battery_drain_rate(self) -> float:
        """Get current battery drain rate based on activity"""
        base_rate = 0.01  # 0.01% per 0.5 seconds for idle
        
        if self.state.status == "moving":
            return base_rate * 3.0
        elif self.state.status == "working":
            return base_rate * 2.0
        elif self.state.status == "charging":
            return -base_rate * 10.0  # Charging
        else:
            return base_rate
    
    async def learning_updater(self):
        """Update learning algorithm periodically"""
        while True:
            await asyncio.sleep(5.0)
            
            # Update exploration rate (epsilon decay)
            self.q_learning.decay_exploration_rate()
            
            # Record learning performance
            convergence = self.q_learning.calculate_convergence()
            self.learning_performance.append({
                'timestamp': time.time(),
                'convergence': convergence,
                'exploration_rate': self.q_learning.exploration_rate,
                'tasks_completed': self.completed_tasks
            })
    
    async def fault_monitor(self):
        """Monitor for robot faults and handle recovery"""
        while True:
            await asyncio.sleep(2.0)
            
            # Check for various fault conditions
            faults = await self.fault_handler.check_robot_health(self.state)
            
            if faults:
                self.logger.warning(f"Faults detected: {faults}")
                await self.handle_faults(faults)
    
    async def handle_faults(self, faults: List[str]):
        """Handle detected faults"""
        for fault in faults:
            if fault == "low_battery":
                await self.handle_low_battery()
            elif fault == "communication_loss":
                await self.handle_communication_loss()
            elif fault == "task_timeout":
                await self.handle_task_timeout()
            else:
                self.logger.warning(f"Unknown fault type: {fault}")
    
    async def handle_low_battery(self):
        """Handle low battery condition"""
        if self.state.battery_level < 20.0:
            self.logger.warning("Low battery detected, seeking charging station")
            
            if self.current_task:
                # Abort current task if battery is critically low
                if self.state.battery_level < 10.0:
                    await self.report_task_completion(self.current_task, False)
                    self.current_task = None
                    self.state.status = "charging"
            
            # Simulate going to charging station
            await self.go_to_charging_station()
    
    async def go_to_charging_station(self):
        """Go to nearest charging station"""
        # In a real implementation, this would use navigation to reach charging station
        self.logger.info("Moving to charging station")
        self.state.status = "charging"
        
        # Simulate charging
        while self.state.battery_level < 90.0:
            self.state.battery_level = min(100.0, self.state.battery_level + 2.0)
            await asyncio.sleep(1.0)
        
        self.logger.info("Charging complete")
        self.state.status = "idle"
    
    async def handle_communication_loss(self):
        """Handle communication loss with master"""
        self.logger.warning("Communication loss detected")
        
        # Attempt to reconnect
        retry_count = 0
        max_retries = 5
        
        while retry_count < max_retries:
            try:
                await self.register_with_master()
                self.logger.info("Communication restored")
                break
            except Exception as e:
                retry_count += 1
                self.logger.warning(f"Reconnection attempt {retry_count} failed: {e}")
                await asyncio.sleep(5.0)
        
        if retry_count >= max_retries:
            self.logger.error("Failed to restore communication, entering autonomous mode")
            # Enter autonomous mode with reduced functionality
    
    async def handle_task_timeout(self):
        """Handle task execution timeout"""
        if self.current_task:
            self.logger.warning(f"Task {self.current_task['task_id']} timed out")
            await self.report_task_completion(self.current_task, False)
            self.current_task = None
            self.state.status = "idle"
    
    async def battery_monitor(self):
        """Monitor battery level and manage power"""
        while True:
            await asyncio.sleep(10.0)
            
            if self.state.battery_level < 25.0 and self.state.status != "charging":
                self.logger.warning(f"Battery level low: {self.state.battery_level:.1f}%")
            
            if self.state.battery_level < 10.0:
                await self.handle_low_battery()
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        uptime = time.time() - self.start_time
        
        return {
            'robot_id': self.robot_id,
            'uptime': uptime,
            'tasks_completed': self.completed_tasks,
            'tasks_failed': self.failed_tasks,
            'success_rate': self.completed_tasks / max(1, self.completed_tasks + self.failed_tasks),
            'total_distance': self.total_distance_traveled,
            'energy_consumed': self.energy_consumed,
            'current_battery': self.state.battery_level,
            'current_position': self.state.position,
            'current_status': self.state.status,
            'learning_convergence': self.q_learning.calculate_convergence() if self.learning_performance else 0.0
        }

async def main():
    """Main entry point for robot agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Robot Agent")
    parser.add_argument("--robot-id", type=str, help="Robot ID")
    parser.add_argument("--config", type=str, default="config/robot_config.yaml", help="Config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Create and start robot agent
    agent = RobotAgent(robot_id=args.robot_id, config_file=args.config)
    
    try:
        await agent.start()
    except KeyboardInterrupt:
        print(f"\nShutting down robot agent {agent.robot_id}...")
        logging.info(f"Robot agent {agent.robot_id} shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
