#!/usr/bin/env python3
"""
Multi-Robot Coordination Master Node
Manages distributed coordination for autonomous robots using reinforcement learning
"""

import argparse
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import yaml
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from algorithms.q_learning import QLearningCoordinator
from algorithms.auction_algorithm import AuctionAllocator
from communication.ros_interface import ROSInterface
from communication.fault_tolerance import FaultToleranceManager
from communication.message_broker import MessageBroker
from utils.config import ConfigManager
from utils.logger import setup_logger
from utils.metrics import MetricsCollector

@dataclass
class Task:
    """Task definition for robot coordination"""
    task_id: str
    task_type: str
    priority: float
    location: Tuple[float, float]
    deadline: float
    required_capabilities: List[str]
    estimated_duration: float
    reward: float = 0.0
    assigned_robot: Optional[str] = None
    status: str = "pending"  # pending, assigned, in_progress, completed, failed

@dataclass
class RobotStatus:
    """Robot status information"""
    robot_id: str
    position: Tuple[float, float]
    capabilities: List[str]
    battery_level: float
    task_load: float
    last_heartbeat: float
    status: str = "active"  # active, busy, maintenance, offline

class CoordinationMaster:
    """Central coordination node for multi-robot system"""
    
    def __init__(self, config_file: str = "config/system_config.yaml"):
        self.config = ConfigManager(config_file)
        self.logger = setup_logger("coordination_master")
        
        # Core components
        self.q_learning = QLearningCoordinator(
            learning_rate=self.config.get("learning.learning_rate", 0.01),
            discount_factor=self.config.get("learning.discount_factor", 0.95),
            exploration_rate=self.config.get("learning.exploration_rate", 0.15)
        )
        
        self.auction_allocator = AuctionAllocator(
            auction_timeout=self.config.get("auction.timeout", 5.0),
            min_bid_threshold=self.config.get("auction.min_bid", 0.1)
        )
        
        self.ros_interface = ROSInterface()
        self.fault_manager = FaultToleranceManager()
        self.message_broker = MessageBroker()
        self.metrics = MetricsCollector()
        
        # State management
        self.robots: Dict[str, RobotStatus] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue = deque()
        self.allocation_history = []
        self.performance_history = deque(maxlen=1000)
        
        # Coordination parameters
        self.max_robots = self.config.get("coordination.max_robots", 50)
        self.heartbeat_interval = self.config.get("coordination.heartbeat_interval", 1.0)
        self.task_timeout = self.config.get("coordination.task_timeout", 30.0)
        
        # Performance tracking
        self.start_time = time.time()
        self.total_tasks_assigned = 0
        self.total_tasks_completed = 0
        self.coordination_efficiency = 0.0
        
        self.logger.info("Coordination Master initialized")
    
    async def start(self):
        """Start the coordination master"""
        self.logger.info("Starting Multi-Robot Coordination Master")
        
        # Initialize ROS interface
        await self.ros_interface.initialize()
        
        # Start core services
        tasks = [
            self.heartbeat_monitor(),
            self.task_processor(),
            self.performance_monitor(),
            self.fault_monitor(),
            self.metrics_collector()
        ]
        
        # Run all services concurrently
        await asyncio.gather(*tasks)
    
    async def register_robot(self, robot_id: str, capabilities: List[str], 
                           position: Tuple[float, float] = (0.0, 0.0)):
        """Register a new robot with the coordination system"""
        if len(self.robots) >= self.max_robots:
            self.logger.warning(f"Maximum robot limit reached: {self.max_robots}")
            return False
        
        robot_status = RobotStatus(
            robot_id=robot_id,
            position=position,
            capabilities=capabilities,
            battery_level=100.0,
            task_load=0.0,
            last_heartbeat=time.time()
        )
        
        self.robots[robot_id] = robot_status
        self.logger.info(f"Robot {robot_id} registered with capabilities: {capabilities}")
        
        # Initialize Q-learning state for new robot
        self.q_learning.add_agent(robot_id)
        
        return True
    
    async def submit_task(self, task: Task):
        """Submit a new task for allocation"""
        self.tasks[task.task_id] = task
        self.task_queue.append(task.task_id)
        
        self.logger.info(f"Task {task.task_id} submitted: {task.task_type} at {task.location}")
        
        # Trigger immediate allocation if robots are available
        await self.allocate_tasks()
    
    async def allocate_tasks(self):
        """Allocate pending tasks to available robots using auction algorithm"""
        if not self.task_queue or not self.robots:
            return
        
        available_robots = [
            robot_id for robot_id, robot in self.robots.items()
            if robot.status == "active" and robot.task_load < 0.8
        ]
        
        if not available_robots:
            return
        
        while self.task_queue and available_robots:
            task_id = self.task_queue.popleft()
            task = self.tasks[task_id]
            
            if task.status != "pending":
                continue
            
            # Run auction for task allocation
            start_time = time.time()
            winning_robot, winning_bid = await self.auction_allocator.run_auction(
                task, available_robots, self.robots
            )
            allocation_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if winning_robot:
                # Assign task to winning robot
                task.assigned_robot = winning_robot
                task.status = "assigned"
                
                # Update robot status
                robot = self.robots[winning_robot]
                robot.task_load += task.estimated_duration / 60.0  # Convert to relative load
                
                # Record allocation
                self.allocation_history.append({
                    'task_id': task_id,
                    'robot_id': winning_robot,
                    'bid': winning_bid,
                    'allocation_time': allocation_time,
                    'timestamp': time.time()
                })
                
                # Send task to robot
                await self.ros_interface.send_task_assignment(winning_robot, task)
                
                # Update Q-learning with reward
                reward = self.calculate_allocation_reward(task, winning_robot)
                self.q_learning.update_q_value(winning_robot, task.task_type, reward)
                
                self.total_tasks_assigned += 1
                self.logger.info(f"Task {task_id} assigned to robot {winning_robot} "
                               f"(bid: {winning_bid:.3f}, time: {allocation_time:.1f}ms)")
                
                # Remove robot from available list if at capacity
                if robot.task_load >= 0.8:
                    available_robots.remove(winning_robot)
            else:
                # No suitable robot found, return task to queue
                self.task_queue.appendleft(task_id)
                break
    
    def calculate_allocation_reward(self, task: Task, robot_id: str) -> float:
        """Calculate reward for task allocation"""
        robot = self.robots[robot_id]
        
        # Distance penalty
        distance = np.sqrt((task.location[0] - robot.position[0])**2 + 
                          (task.location[1] - robot.position[1])**2)
        distance_reward = max(0, 1.0 - distance / 100.0)  # Normalize to 100m range
        
        # Capability match bonus
        capability_match = len(set(task.required_capabilities) & set(robot.capabilities))
        capability_reward = capability_match / max(1, len(task.required_capabilities))
        
        # Load balancing penalty
        load_penalty = robot.task_load * 0.5
        
        # Priority bonus
        priority_bonus = task.priority * 0.3
        
        total_reward = distance_reward + capability_reward + priority_bonus - load_penalty
        return max(0.1, total_reward)  # Minimum reward threshold
    
    async def heartbeat_monitor(self):
        """Monitor robot heartbeats and detect failures"""
        while True:
            current_time = time.time()
            offline_robots = []
            
            for robot_id, robot in self.robots.items():
                time_since_heartbeat = current_time - robot.last_heartbeat
                
                if time_since_heartbeat > self.heartbeat_interval * 3:
                    if robot.status != "offline":
                        self.logger.warning(f"Robot {robot_id} appears offline "
                                          f"(last heartbeat: {time_since_heartbeat:.1f}s ago)")
                        robot.status = "offline"
                        offline_robots.append(robot_id)
                        
                        # Trigger fault recovery
                        await self.fault_manager.handle_robot_failure(robot_id, self.tasks)
            
            if offline_robots:
                await self.reallocate_failed_tasks(offline_robots)
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def reallocate_failed_tasks(self, failed_robots: List[str]):
        """Reallocate tasks from failed robots"""
        for robot_id in failed_robots:
            # Find tasks assigned to failed robot
            failed_tasks = [
                task for task in self.tasks.values()
                if task.assigned_robot == robot_id and task.status in ["assigned", "in_progress"]
            ]
            
            for task in failed_tasks:
                task.assigned_robot = None
                task.status = "pending"
                self.task_queue.append(task.task_id)
                
                self.logger.info(f"Reallocating task {task.task_id} due to robot {robot_id} failure")
        
        # Trigger reallocation
        await self.allocate_tasks()
    
    async def task_processor(self):
        """Process task queue and handle task lifecycle"""
        while True:
            # Clean up completed/failed tasks
            current_time = time.time()
            expired_tasks = []
            
            for task_id, task in self.tasks.items():
                if task.status == "assigned" and current_time - task.deadline > self.task_timeout:
                    expired_tasks.append(task_id)
                    self.logger.warning(f"Task {task_id} expired")
            
            # Handle expired tasks
            for task_id in expired_tasks:
                task = self.tasks[task_id]
                if task.assigned_robot:
                    robot = self.robots.get(task.assigned_robot)
                    if robot:
                        robot.task_load = max(0, robot.task_load - task.estimated_duration / 60.0)
                
                task.status = "failed"
                # Negative reward for timeout
                if task.assigned_robot:
                    self.q_learning.update_q_value(task.assigned_robot, task.task_type, -0.5)
            
            # Attempt allocation for pending tasks
            if self.task_queue:
                await self.allocate_tasks()
            
            await asyncio.sleep(1.0)
    
    async def handle_task_completion(self, robot_id: str, task_id: str, success: bool):
        """Handle task completion notification from robot"""
        if task_id not in self.tasks:
            self.logger.warning(f"Unknown task completion: {task_id}")
            return
        
        task = self.tasks[task_id]
        robot = self.robots.get(robot_id)
        
        if robot:
            # Update robot load
            robot.task_load = max(0, robot.task_load - task.estimated_duration / 60.0)
        
        if success:
            task.status = "completed"
            self.total_tasks_completed += 1
            
            # Positive reward for successful completion
            completion_reward = 1.0 + task.priority * 0.5
            self.q_learning.update_q_value(robot_id, task.task_type, completion_reward)
            
            self.logger.info(f"Task {task_id} completed successfully by robot {robot_id}")
        else:
            task.status = "failed"
            
            # Small negative reward for failure
            self.q_learning.update_q_value(robot_id, task.task_type, -0.2)
            
            self.logger.warning(f"Task {task_id} failed on robot {robot_id}")
        
        # Update performance metrics
        self.update_performance_metrics()
    
    async def performance_monitor(self):
        """Monitor and log system performance"""
        while True:
            await asyncio.sleep(10.0)  # Update every 10 seconds
            
            metrics = self.calculate_performance_metrics()
            self.performance_history.append(metrics)
            
            if len(self.performance_history) % 6 == 0:  # Log every minute
                self.logger.info(f"Performance: Efficiency={metrics['efficiency']:.3f}, "
                               f"Allocation Time={metrics['avg_allocation_time']:.1f}ms, "
                               f"Active Robots={metrics['active_robots']}")
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate current performance metrics"""
        current_time = time.time()
        
        # Calculate efficiency
        if self.total_tasks_assigned > 0:
            self.coordination_efficiency = self.total_tasks_completed / self.total_tasks_assigned
        else:
            self.coordination_efficiency = 0.0
        
        # Calculate average allocation time
        recent_allocations = [
            alloc for alloc in self.allocation_history
            if current_time - alloc['timestamp'] < 300  # Last 5 minutes
        ]
        
        avg_allocation_time = 0.0
        if recent_allocations:
            avg_allocation_time = np.mean([alloc['allocation_time'] for alloc in recent_allocations])
        
        # Count active robots
        active_robots = sum(1 for robot in self.robots.values() if robot.status == "active")
        
        # Calculate convergence metrics
        q_convergence = self.q_learning.calculate_convergence()
        policy_gradient = self.q_learning.calculate_policy_gradient()
        
        return {
            'timestamp': current_time,
            'efficiency': self.coordination_efficiency,
            'avg_allocation_time': avg_allocation_time,
            'active_robots': active_robots,
            'total_robots': len(self.robots),
            'pending_tasks': len(self.task_queue),
            'q_convergence': q_convergence,
            'policy_gradient': policy_gradient,
            'tasks_assigned': self.total_tasks_assigned,
            'tasks_completed': self.total_tasks_completed
        }
    
    def update_performance_metrics(self):
        """Update performance metrics after task completion"""
        # This is called after each task completion
        pass
    
    async def fault_monitor(self):
        """Monitor system health and handle faults"""
        while True:
            await asyncio.sleep(5.0)
            
            # Check system health
            health_status = await self.fault_manager.check_system_health(self.robots)
            
            if health_status['critical_issues']:
                self.logger.error(f"Critical system issues detected: {health_status['critical_issues']}")
                # Implement emergency protocols if needed
            
            # Check for communication issues
            comm_issues = await self.message_broker.check_communication_health()
            if comm_issues:
                self.logger.warning(f"Communication issues: {comm_issues}")
    
    async def metrics_collector(self):
        """Collect and store performance metrics"""
        while True:
            await asyncio.sleep(30.0)  # Collect every 30 seconds
            
            metrics = self.calculate_performance_metrics()
            await self.metrics.record_metrics(metrics)
    
    async def update_robot_heartbeat(self, robot_id: str, position: Tuple[float, float], 
                                   battery_level: float):
        """Update robot heartbeat and status"""
        if robot_id in self.robots:
            robot = self.robots[robot_id]
            robot.last_heartbeat = time.time()
            robot.position = position
            robot.battery_level = battery_level
            
            # Update status based on battery level
            if battery_level < 20.0 and robot.status == "active":
                robot.status = "maintenance"
                self.logger.warning(f"Robot {robot_id} low battery: {battery_level}%")
            elif battery_level > 30.0 and robot.status == "maintenance":
                robot.status = "active"
                self.logger.info(f"Robot {robot_id} battery restored: {battery_level}%")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'robots': {robot_id: asdict(robot) for robot_id, robot in self.robots.items()},
            'tasks': {task_id: asdict(task) for task_id, task in self.tasks.items()},
            'metrics': self.calculate_performance_metrics(),
            'uptime': time.time() - self.start_time
        }

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Multi-Robot Coordination Master")
    parser.add_argument("--robots", type=int, default=5, help="Expected number of robots")
    parser.add_argument("--environment", type=str, default="warehouse", help="Environment type")
    parser.add_argument("--config", type=str, default="config/system_config.yaml", help="Config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start coordination master
    master = CoordinationMaster(args.config)
    
    try:
        await master.start()
    except KeyboardInterrupt:
        print("\nShutting down coordination master...")
        logging.info("Coordination master shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
