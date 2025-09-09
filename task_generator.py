#!/usr/bin/env python3
"""
Task Generator for Multi-Robot Coordination Framework
Generates realistic tasks for testing and simulation
"""

import asyncio
import logging
import time
import random
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import json

from utils.config import ConfigManager
from utils.logger import setup_logger
from communication.ros_interface import ROSInterface

@dataclass
class Task:
    """Task definition"""
    task_id: str
    task_type: str
    priority: float
    location: Tuple[float, float]
    deadline: float
    required_capabilities: List[str]
    estimated_duration: float
    reward: float = 0.0
    metadata: Dict = None

class TaskGenerator:
    """Generates tasks for robot coordination system"""
    
    def __init__(self, config_file: str = "config/system_config.yaml"):
        self.config = ConfigManager(config_file)
        self.logger = setup_logger("task_generator")
        
        # Task generation parameters
        self.task_rate = 0.5  # tasks per second
        self.complexity = "medium"  # low, medium, high
        self.environment_type = "warehouse"
        
        # Environment boundaries
        self.boundaries = self.config.get("environment.boundaries", {
            'x_min': -50.0, 'x_max': 50.0,
            'y_min': -50.0, 'y_max': 50.0
        })
        
        # Task types and their properties
        self.task_types = {
            'navigation': {
                'capabilities': ['navigation'],
                'duration_range': (10, 30),
                'priority_range': (0.5, 1.5),
                'complexity_factor': 1.0
            },
            'pickup': {
                'capabilities': ['navigation', 'manipulation'],
                'duration_range': (15, 45),
                'priority_range': (1.0, 2.0),
                'complexity_factor': 1.2
            },
            'delivery': {
                'capabilities': ['navigation', 'manipulation'],
                'duration_range': (20, 60),
                'priority_range': (1.2, 2.5),
                'complexity_factor': 1.3
            },
            'inspection': {
                'capabilities': ['navigation', 'sensing'],
                'duration_range': (25, 40),
                'priority_range': (0.8, 1.8),
                'complexity_factor': 1.1
            },
            'cleaning': {
                'capabilities': ['navigation', 'cleaning'],
                'duration_range': (30, 90),
                'priority_range': (0.3, 1.0),
                'complexity_factor': 0.9
            },
            'security_patrol': {
                'capabilities': ['navigation', 'sensing', 'security'],
                'duration_range': (60, 180),
                'priority_range': (1.5, 3.0),
                'complexity_factor': 1.4
            },
            'maintenance': {
                'capabilities': ['navigation', 'manipulation', 'tools'],
                'duration_range': (45, 120),
                'priority_range': (2.0, 3.5),
                'complexity_factor': 1.6
            },
            'sorting': {
                'capabilities': ['manipulation', 'sensing'],
                'duration_range': (20, 50),
                'priority_range': (1.0, 2.0),
                'complexity_factor': 1.1
            },
            'transport': {
                'capabilities': ['navigation', 'transport'],
                'duration_range': (40, 100),
                'priority_range': (1.3, 2.2),
                'complexity_factor': 1.2
            },
            'emergency_response': {
                'capabilities': ['navigation', 'emergency', 'communication'],
                'duration_range': (5, 20),
                'priority_range': (3.0, 5.0),
                'complexity_factor': 2.0
            }
        }
        
        # Communication
        self.ros_interface = ROSInterface("task_generator")
        
        # Task tracking
        self.generated_tasks = []
        self.task_counter = 0
        self.start_time = time.time()
        
        # Statistics
        self.tasks_generated = 0
        self.tasks_by_type = {task_type: 0 for task_type in self.task_types.keys()}
        
        self.logger.info("Task generator initialized")
    
    async def start_generation(self):
        """Start generating tasks"""
        self.logger.info(f"Starting task generation at {self.task_rate} tasks/second")
        
        # Initialize ROS interface
        await self.ros_interface.initialize()
        
        # Start generation loop
        while True:
            try:
                # Generate task
                task = self.generate_task()
                
                # Submit task to coordination system
                await self.submit_task(task)
                
                # Update statistics
                self.tasks_generated += 1
                self.tasks_by_type[task.task_type] += 1
                
                # Log progress
                if self.tasks_generated % 10 == 0:
                    self.logger.info(f"Generated {self.tasks_generated} tasks")
                
                # Wait for next task
                interval = 1.0 / self.task_rate
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error generating task: {e}")
                await asyncio.sleep(1.0)
    
    def generate_task(self) -> Task:
        """Generate a single task"""
        # Select task type based on complexity and environment
        task_type = self.select_task_type()
        
        # Generate task parameters
        task_id = f"task_{self.task_counter:06d}"
        self.task_counter += 1
        
        location = self.generate_location(task_type)
        priority = self.generate_priority(task_type)
        deadline = self.generate_deadline(task_type)
        capabilities = self.task_types[task_type]['capabilities'].copy()
        duration = self.generate_duration(task_type)
        reward = self.calculate_reward(task_type, priority, duration)
        
        # Create metadata
        metadata = {
            'generated_at': time.time(),
            'environment': self.environment_type,
            'complexity': self.complexity,
            'generator_id': 'task_generator'
        }
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            location=location,
            deadline=deadline,
            required_capabilities=capabilities,
            estimated_duration=duration,
            reward=reward,
            metadata=metadata
        )
        
        self.generated_tasks.append(task)
        self.logger.debug(f"Generated task: {task_id} ({task_type}) at {location}")
        
        return task
    
    def select_task_type(self) -> str:
        """Select task type based on environment and complexity"""
        # Weight task types based on environment
        if self.environment_type == "warehouse":
            weights = {
                'pickup': 0.25,
                'delivery': 0.25,
                'transport': 0.15,
                'sorting': 0.10,
                'inspection': 0.10,
                'navigation': 0.05,
                'cleaning': 0.05,
                'maintenance': 0.03,
                'security_patrol': 0.02
            }
        elif self.environment_type == "factory":
            weights = {
                'maintenance': 0.30,
                'inspection': 0.25,
                'transport': 0.15,
                'sorting': 0.10,
                'pickup': 0.08,
                'delivery': 0.07,
                'cleaning': 0.03,
                'security_patrol': 0.02
            }
        elif self.environment_type == "hospital":
            weights = {
                'delivery': 0.35,
                'transport': 0.25,
                'cleaning': 0.15,
                'inspection': 0.10,
                'navigation': 0.08,
                'security_patrol': 0.05,
                'emergency_response': 0.02
            }
        else:  # outdoor/general
            weights = {
                'navigation': 0.30,
                'inspection': 0.20,
                'security_patrol': 0.15,
                'maintenance': 0.10,
                'pickup': 0.10,
                'delivery': 0.08,
                'emergency_response': 0.05,
                'transport': 0.02
            }
        
        # Adjust weights based on complexity
        if self.complexity == "low":
            # Favor simpler tasks
            for task_type in weights:
                complexity_factor = self.task_types[task_type]['complexity_factor']
                if complexity_factor > 1.2:
                    weights[task_type] *= 0.5
        elif self.complexity == "high":
            # Favor complex tasks
            for task_type in weights:
                complexity_factor = self.task_types[task_type]['complexity_factor']
                if complexity_factor > 1.2:
                    weights[task_type] *= 1.5
        
        # Select based on weights
        task_types = list(weights.keys())
        weight_values = list(weights.values())
        
        return np.random.choice(task_types, p=weight_values)
    
    def generate_location(self, task_type: str) -> Tuple[float, float]:
        """Generate task location based on type and environment"""
        # Basic random location within boundaries
        x = np.random.uniform(self.boundaries['x_min'], self.boundaries['x_max'])
        y = np.random.uniform(self.boundaries['y_min'], self.boundaries['y_max'])
        
        # Add task-specific location patterns
        if task_type in ['pickup', 'delivery']:
            # Cluster near storage areas
            if np.random.random() < 0.7:
                storage_centers = [(20, 20), (-20, -20), (0, 30)]
                center = random.choice(storage_centers)
                x = np.random.normal(center[0], 5)
                y = np.random.normal(center[1], 5)
        
        elif task_type == 'security_patrol':
            # Patrol routes along perimeter
            if np.random.random() < 0.6:
                # Generate points along boundary
                side = random.choice(['top', 'bottom', 'left', 'right'])
                if side == 'top':
                    y = self.boundaries['y_max'] - 5
                    x = np.random.uniform(self.boundaries['x_min'], self.boundaries['x_max'])
                elif side == 'bottom':
                    y = self.boundaries['y_min'] + 5
                    x = np.random.uniform(self.boundaries['x_min'], self.boundaries['x_max'])
                elif side == 'left':
                    x = self.boundaries['x_min'] + 5
                    y = np.random.uniform(self.boundaries['y_min'], self.boundaries['y_max'])
                else:  # right
                    x = self.boundaries['x_max'] - 5
                    y = np.random.uniform(self.boundaries['y_min'], self.boundaries['y_max'])
        
        elif task_type == 'cleaning':
            # Cluster in high-traffic areas
            high_traffic = [(0, 0), (15, 15), (-15, -15)]
            if np.random.random() < 0.8:
                center = random.choice(high_traffic)
                x = np.random.normal(center[0], 8)
                y = np.random.normal(center[1], 8)
        
        # Ensure location is within boundaries
        x = np.clip(x, self.boundaries['x_min'], self.boundaries['x_max'])
        y = np.clip(y, self.boundaries['y_min'], self.boundaries['y_max'])
        
        return (float(x), float(y))
    
    def generate_priority(self, task_type: str) -> float:
        """Generate task priority"""
        priority_range = self.task_types[task_type]['priority_range']
        base_priority = np.random.uniform(priority_range[0], priority_range[1])
        
        # Add time-based priority variations
        current_time = time.time()
        time_of_day = (current_time % 86400) / 86400  # 0-1 for 24 hours
        
        # Higher priority during business hours (8 AM - 6 PM)
        if 0.33 < time_of_day < 0.75:  # Business hours
            base_priority *= 1.2
        
        # Emergency tasks get random high priority spikes
        if task_type == 'emergency_response':
            if np.random.random() < 0.3:  # 30% chance of critical emergency
                base_priority *= 2.0
        
        # Complexity adjustment
        if self.complexity == "high":
            base_priority *= 1.1
        elif self.complexity == "low":
            base_priority *= 0.9
        
        return float(base_priority)
    
    def generate_deadline(self, task_type: str) -> float:
        """Generate task deadline"""
        current_time = time.time()
        
        # Base deadline based on task type
        if task_type == 'emergency_response':
            deadline_offset = np.random.uniform(60, 300)  # 1-5 minutes
        elif task_type in ['pickup', 'delivery']:
            deadline_offset = np.random.uniform(600, 1800)  # 10-30 minutes
        elif task_type == 'security_patrol':
            deadline_offset = np.random.uniform(1800, 3600)  # 30-60 minutes
        elif task_type == 'maintenance':
            deadline_offset = np.random.uniform(3600, 7200)  # 1-2 hours
        else:
            deadline_offset = np.random.uniform(900, 2700)  # 15-45 minutes
        
        # Add some randomness
        deadline_offset *= np.random.uniform(0.8, 1.2)
        
        return current_time + deadline_offset
    
    def generate_duration(self, task_type: str) -> float:
        """Generate estimated task duration"""
        duration_range = self.task_types[task_type]['duration_range']
        base_duration = np.random.uniform(duration_range[0], duration_range[1])
        
        # Complexity adjustment
        if self.complexity == "high":
            base_duration *= 1.3
        elif self.complexity == "low":
            base_duration *= 0.8
        
        # Add some variability
        base_duration *= np.random.uniform(0.9, 1.1)
        
        return float(base_duration)
    
    def calculate_reward(self, task_type: str, priority: float, duration: float) -> float:
        """Calculate task reward"""
        # Base reward based on task complexity
        complexity_factor = self.task_types[task_type]['complexity_factor']
        base_reward = complexity_factor * 10.0
        
        # Priority multiplier
        priority_multiplier = 1.0 + (priority - 1.0) * 0.5
        
        # Duration bonus (longer tasks get higher rewards)
        duration_bonus = 1.0 + (duration / 60.0) * 0.1  # 10% per minute
        
        total_reward = base_reward * priority_multiplier * duration_bonus
        
        # Add small random factor
        total_reward *= np.random.uniform(0.95, 1.05)
        
        return float(total_reward)
    
    async def submit_task(self, task: Task):
        """Submit task to coordination system"""
        try:
            # Prepare task data for transmission
            task_data = {
                'task_id': task.task_id,
                'task_type': task.task_type,
                'priority': task.priority,
                'location': task.location,
                'deadline': task.deadline,
                'required_capabilities': task.required_capabilities,
                'estimated_duration': task.estimated_duration,
                'reward': task.reward,
                'metadata': task.metadata
            }
            
            # Send via ROS interface
            success = await self.ros_interface.publish_message('coordination', {
                'sender_id': 'task_generator',
                'receiver_id': 'coordination_master',
                'message_type': 'new_task',
                'timestamp': time.time(),
                'data': task_data,
                'message_id': f"new_task_{task.task_id}",
                'priority': min(int(task.priority), 5)
            })
            
            if success:
                self.logger.debug(f"Submitted task {task.task_id} successfully")
            else:
                self.logger.warning(f"Failed to submit task {task.task_id}")
            
        except Exception as e:
            self.logger.error(f"Error submitting task {task.task_id}: {e}")
    
    def set_generation_rate(self, rate: float):
        """Set task generation rate"""
        self.task_rate = max(0.01, min(10.0, rate))  # Clamp between 0.01 and 10 Hz
        self.logger.info(f"Task generation rate set to {self.task_rate} tasks/second")
    
    def set_complexity(self, complexity: str):
        """Set task complexity level"""
        if complexity in ['low', 'medium', 'high']:
            self.complexity = complexity
            self.logger.info(f"Task complexity set to {complexity}")
        else:
            self.logger.warning(f"Invalid complexity level: {complexity}")
    
    def set_environment(self, environment: str):
        """Set environment type"""
        if environment in ['warehouse', 'factory', 'hospital', 'outdoor']:
            self.environment_type = environment
            self.logger.info(f"Environment type set to {environment}")
        else:
            self.logger.warning(f"Invalid environment type: {environment}")
    
    async def generate_batch_tasks(self, count: int) -> List[Task]:
        """Generate a batch of tasks"""
        tasks = []
        
        for _ in range(count):
            task = self.generate_task()
            tasks.append(task)
            
            # Submit task
            await self.submit_task(task)
            
            # Small delay between tasks
            await asyncio.sleep(0.1)
        
        self.logger.info(f"Generated batch of {count} tasks")
        return tasks
    
    async def generate_scenario_tasks(self, scenario: str) -> List[Task]:
        """Generate tasks for specific scenarios"""
        tasks = []
        
        if scenario == "emergency":
            # Generate emergency scenario with high-priority tasks
            for i in range(3):
                # Override normal generation for emergency
                old_complexity = self.complexity
                self.complexity = "high"
                
                task = self.generate_task()
                task.task_type = "emergency_response"
                task.priority = np.random.uniform(4.0, 5.0)
                task.deadline = time.time() + 300  # 5 minutes
                task.required_capabilities = ['navigation', 'emergency', 'communication']
                
                tasks.append(task)
                await self.submit_task(task)
                
                self.complexity = old_complexity
        
        elif scenario == "maintenance_day":
            # Generate multiple maintenance tasks
            for i in range(5):
                task = self.generate_task()
                task.task_type = "maintenance"
                task.priority = np.random.uniform(2.0, 3.0)
                task.deadline = time.time() + 7200  # 2 hours
                
                tasks.append(task)
                await self.submit_task(task)
        
        elif scenario == "busy_warehouse":
            # Generate high volume of warehouse tasks
            task_types = ['pickup', 'delivery', 'transport', 'sorting']
            for i in range(10):
                task = self.generate_task()
                task.task_type = random.choice(task_types)
                task.priority = np.random.uniform(1.5, 2.5)
                
                tasks.append(task)
                await self.submit_task(task)
        
        elif scenario == "security_alert":
            # Generate security patrol tasks
            for i in range(4):
                task = self.generate_task()
                task.task_type = "security_patrol"
                task.priority = np.random.uniform(2.5, 4.0)
                task.deadline = time.time() + 1800  # 30 minutes
                
                tasks.append(task)
                await self.submit_task(task)
        
        self.logger.info(f"Generated {len(tasks)} tasks for scenario: {scenario}")
        return tasks
    
    def get_statistics(self) -> Dict:
        """Get task generation statistics"""
        uptime = time.time() - self.start_time
        
        return {
            'total_tasks_generated': self.tasks_generated,
            'generation_rate': self.task_rate,
            'uptime_seconds': uptime,
            'actual_rate': self.tasks_generated / max(uptime, 1),
            'tasks_by_type': self.tasks_by_type.copy(),
            'complexity': self.complexity,
            'environment': self.environment_type,
            'most_common_task': max(self.tasks_by_type, key=self.tasks_by_type.get) if self.tasks_by_type else None
        }
    
    async def adaptive_generation(self):
        """Adapt task generation based on system state"""
        while True:
            try:
                # Get system status (would integrate with system monitor)
                # For now, simulate adaptive behavior
                
                current_time = time.time()
                time_of_day = (current_time % 86400) / 86400
                
                # Adjust generation rate based on time of day
                if 0.33 < time_of_day < 0.75:  # Business hours
                    target_rate = self.task_rate * 1.5
                elif 0.75 < time_of_day < 0.9:  # Evening
                    target_rate = self.task_rate * 0.8
                else:  # Night
                    target_rate = self.task_rate * 0.3
                
                # Smooth rate adjustment
                self.task_rate += (target_rate - self.task_rate) * 0.1
                
                # Randomly trigger scenarios
                if np.random.random() < 0.001:  # 0.1% chance per check
                    scenarios = ["emergency", "maintenance_day", "busy_warehouse", "security_alert"]
                    scenario = random.choice(scenarios)
                    self.logger.info(f"Triggering adaptive scenario: {scenario}")
                    await self.generate_scenario_tasks(scenario)
                
            except Exception as e:
                self.logger.error(f"Error in adaptive generation: {e}")
            
            await asyncio.sleep(60.0)  # Check every minute
    
    async def export_generated_tasks(self, filename: str = None):
        """Export generated tasks to file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"generated_tasks_{timestamp}.json"
        
        try:
            export_data = {
                'metadata': {
                    'export_time': time.time(),
                    'total_tasks': len(self.generated_tasks),
                    'generation_rate': self.task_rate,
                    'complexity': self.complexity,
                    'environment': self.environment_type
                },
                'statistics': self.get_statistics(),
                'tasks': []
            }
            
            # Convert tasks to serializable format
            for task in self.generated_tasks:
                task_dict = {
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'priority': task.priority,
                    'location': task.location,
                    'deadline': task.deadline,
                    'required_capabilities': task.required_capabilities,
                    'estimated_duration': task.estimated_duration,
                    'reward': task.reward,
                    'metadata': task.metadata
                }
                export_data['tasks'].append(task_dict)
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported {len(self.generated_tasks)} tasks to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting tasks: {e}")
    
    async def load_task_template(self, template_file: str):
        """Load task generation template from file"""
        try:
            with open(template_file, 'r') as f:
                template = json.load(f)
            
            # Update task types if provided
            if 'task_types' in template:
                self.task_types.update(template['task_types'])
            
            # Update generation parameters
            if 'generation_params' in template:
                params = template['generation_params']
                self.task_rate = params.get('task_rate', self.task_rate)
                self.complexity = params.get('complexity', self.complexity)
                self.environment_type = params.get('environment', self.environment_type)
            
            # Update boundaries
            if 'boundaries' in template:
                self.boundaries.update(template['boundaries'])
            
            self.logger.info(f"Loaded task template from {template_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading task template: {e}")


class InteractiveTaskGenerator:
    """Interactive task generator with command interface"""
    
    def __init__(self, generator: TaskGenerator):
        self.generator = generator
        self.logger = logging.getLogger("interactive_task_generator")
        self.running = False
    
    async def start_interactive_mode(self):
        """Start interactive command mode"""
        self.running = True
        self.logger.info("Starting interactive task generator mode")
        
        print("\n=== Interactive Task Generator ===")
        print("Commands:")
        print("  generate <count>     - Generate <count> tasks")
        print("  scenario <name>      - Generate scenario tasks")
        print("  rate <value>         - Set generation rate")
        print("  complexity <level>   - Set complexity (low/medium/high)")
        print("  environment <type>   - Set environment type")
        print("  stats                - Show statistics")
        print("  export               - Export tasks to file")
        print("  help                 - Show this help")
        print("  quit                 - Exit interactive mode")
        print()
        
        while self.running:
            try:
                command = input("task_gen> ").strip().split()
                
                if not command:
                    continue
                
                await self.process_command(command)
                
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def process_command(self, command: List[str]):
        """Process interactive command"""
        cmd = command[0].lower()
        
        if cmd == "generate":
            count = int(command[1]) if len(command) > 1 else 1
            tasks = await self.generator.generate_batch_tasks(count)
            print(f"Generated {len(tasks)} tasks")
        
        elif cmd == "scenario":
            scenario = command[1] if len(command) > 1 else "emergency"
            tasks = await self.generator.generate_scenario_tasks(scenario)
            print(f"Generated {len(tasks)} tasks for scenario: {scenario}")
        
        elif cmd == "rate":
            rate = float(command[1]) if len(command) > 1 else 0.5
            self.generator.set_generation_rate(rate)
            print(f"Generation rate set to {rate} tasks/second")
        
        elif cmd == "complexity":
            level = command[1] if len(command) > 1 else "medium"
            self.generator.set_complexity(level)
            print(f"Complexity set to {level}")
        
        elif cmd == "environment":
            env_type = command[1] if len(command) > 1 else "warehouse"
            self.generator.set_environment(env_type)
            print(f"Environment set to {env_type}")
        
        elif cmd == "stats":
            stats = self.generator.get_statistics()
            print("\n=== Task Generation Statistics ===")
            for key, value in stats.items():
                print(f"{key}: {value}")
            print()
        
        elif cmd == "export":
            filename = command[1] if len(command) > 1 else None
            await self.generator.export_generated_tasks(filename)
            print("Tasks exported successfully")
        
        elif cmd == "help":
            print("\nAvailable commands:")
            print("  generate <count>     - Generate <count> tasks")
            print("  scenario <name>      - Generate scenario tasks (emergency, maintenance_day, busy_warehouse, security_alert)")
            print("  rate <value>         - Set generation rate (tasks per second)")
            print("  complexity <level>   - Set complexity (low/medium/high)")
            print("  environment <type>   - Set environment (warehouse/factory/hospital/outdoor)")
            print("  stats                - Show generation statistics")
            print("  export [filename]    - Export tasks to JSON file")
            print("  quit                 - Exit interactive mode")
            print()
        
        elif cmd == "quit":
            self.running = False
            print("Exiting interactive mode...")
        
        else:
            print(f"Unknown command: {cmd}. Type 'help' for available commands.")


async def main():
    """Main entry point for task generator"""
    parser = argparse.ArgumentParser(description="Multi-Robot Task Generator")
    parser.add_argument("--rate", type=float, default=0.5, help="Task generation rate (tasks/second)")
    parser.add_argument("--complexity", type=str, default="medium", choices=["low", "medium", "high"], help="Task complexity")
    parser.add_argument("--environment", type=str, default="warehouse", choices=["warehouse", "factory", "hospital", "outdoor"], help="Environment type")
    parser.add_argument("--config", type=str, default="config/system_config.yaml", help="Config file")
    parser.add_argument("--interactive", action="store_true", help="Start in interactive mode")
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive generation")
    parser.add_argument("--scenario", type=str, help="Generate specific scenario")
    parser.add_argument("--count", type=int, help="Generate specific number of tasks then exit")
    parser.add_argument("--template", type=str, help="Load task template file")
    
    args = parser.parse_args()
    
    # Create task generator
    generator = TaskGenerator(args.config)
    
    # Set parameters
    generator.set_generation_rate(args.rate)
    generator.set_complexity(args.complexity)
    generator.set_environment(args.environment)
    
    # Load template if specified
    if args.template:
        await generator.load_task_template(args.template)
    
    try:
        if args.interactive:
            # Interactive mode
            interactive = InteractiveTaskGenerator(generator)
            await interactive.start_interactive_mode()
        
        elif args.scenario:
            # Generate scenario and exit
            await generator.generate_scenario_tasks(args.scenario)
            print(f"Generated scenario: {args.scenario}")
        
        elif args.count:
            # Generate specific count and exit
            tasks = await generator.generate_batch_tasks(args.count)
            print(f"Generated {len(tasks)} tasks")
            
            # Export tasks
            await generator.export_generated_tasks()
        
        else:
            # Continuous generation mode
            tasks = []
            
            # Start adaptive generation if enabled
            if args.adaptive:
                adaptive_task = asyncio.create_task(generator.adaptive_generation())
                tasks.append(adaptive_task)
            
            # Start main generation
            generation_task = asyncio.create_task(generator.start_generation())
            tasks.append(generation_task)
            
            # Run until interrupted
            await asyncio.gather(*tasks)
    
    except KeyboardInterrupt:
        print("\nShutting down task generator...")
        
        # Export final statistics
        stats = generator.get_statistics()
        print(f"\nFinal Statistics:")
        print(f"Total tasks generated: {stats['total_tasks_generated']}")
        print(f"Actual generation rate: {stats['actual_rate']:.3f} tasks/second")
        print(f"Most common task type: {stats['most_common_task']}")
        
        # Export tasks
        await generator.export_generated_tasks()
        
        logging.info("Task generator shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
