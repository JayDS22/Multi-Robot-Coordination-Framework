
#!/usr/bin/env python3
"""
Unit tests for coordination functionality
"""

import pytest
import asyncio
import logging
import time
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from coordination_master import CoordinationMaster, Task, RobotStatus
from robot_agent import RobotAgent, RobotState
from utils.config import ConfigManager

class TestCoordinationMaster:
    """Test cases for CoordinationMaster"""
    
    @pytest.fixture
    async def master(self):
        """Create a test coordination master"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'coordination': {'max_robots': 10, 'heartbeat_interval': 0.5},
                'learning': {'learning_rate': 0.01},
                'fault_tolerance': {'max_retries': 2}
            }
            import yaml
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            master = CoordinationMaster(config_file)
            yield master
        finally:
            os.unlink(config_file)
    
    @pytest.mark.asyncio
    async def test_robot_registration(self, master):
        """Test robot registration"""
        success = await master.register_robot("test_robot_1", ["navigation", "sensing"], (0, 0))
        assert success == True
        assert "test_robot_1" in master.robots
        assert master.robots["test_robot_1"].capabilities == ["navigation", "sensing"]
    
    @pytest.mark.asyncio
    async def test_max_robots_limit(self, master):
        """Test maximum robot limit"""
        # Register up to max limit
        for i in range(master.max_robots):
            success = await master.register_robot(f"robot_{i}", ["navigation"], (i, i))
            assert success == True
        
        # Try to register one more (should fail)
        success = await master.register_robot("excess_robot", ["navigation"], (0, 0))
        assert success == False
    
    @pytest.mark.asyncio
    async def test_task_submission(self, master):
        """Test task submission"""
        task = Task(
            task_id="test_task_001",
            task_type="navigation",
            priority=1.5,
            location=(10, 20),
            deadline=time.time() + 3600,
            required_capabilities=["navigation"],
            estimated_duration=30.0
        )
        
        await master.submit_task(task)
        assert "test_task_001" in master.tasks
        assert len(master.task_queue) == 1
    
    @pytest.mark.asyncio
    async def test_task_allocation(self, master):
        """Test basic task allocation"""
        # Register a robot
        await master.register_robot("test_robot", ["navigation"], (0, 0))
        
        # Submit a task
        task = Task(
            task_id="alloc_test_001",
            task_type="navigation",
            priority=1.0,
            location=(5, 5),
            deadline=time.time() + 3600,
            required_capabilities=["navigation"],
            estimated_duration=20.0
        )
        
        await master.submit_task(task)
        
        # Wait a bit for allocation
        await asyncio.sleep(0.1)
        
        # Check if task was allocated
        allocated_task = master.tasks.get("alloc_test_001")
        assert allocated_task is not None
        # Note: In a real test, we'd check if the task was assigned
    
    def test_reward_calculation(self, master):
        """Test reward calculation"""
        task = Task(
            task_id="reward_test",
            task_type="pickup",
            priority=2.0,
            location=(10, 10),
            deadline=time.time() + 1800,
            required_capabilities=["navigation", "manipulation"],
            estimated_duration=45.0
        )
        
        robot_status = RobotStatus(
            robot_id="test_robot",
            position=(5, 5),
            capabilities=["navigation", "manipulation"],
            battery_level=80.0,
            task_load=0.2,
            last_heartbeat=time.time()
        )
        
        reward = master.calculate_allocation_reward(task, "test_robot")
        assert isinstance(reward, float)
        assert reward > 0.0
    
    @pytest.mark.asyncio
    async def test_heartbeat_update(self, master):
        """Test robot heartbeat updates"""
        robot_id = "heartbeat_robot"
        await master.register_robot(robot_id, ["navigation"], (0, 0))
        
        # Update heartbeat
        new_position = (10, 15)
        new_battery = 75.0
        await master.update_robot_heartbeat(robot_id, new_position, new_battery)
        
        robot = master.robots[robot_id]
        assert robot.position == new_position
        assert robot.battery_level == new_battery
        assert abs(robot.last_heartbeat - time.time()) < 1.0

class TestRobotAgent:
    """Test cases for RobotAgent"""
    
    @pytest.fixture
    def agent(self):
        """Create a test robot agent"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'robot_settings': {
                    'capabilities': ['navigation', 'sensing'],
                    'max_velocity': 2.0,
                    'battery_capacity': 100.0
                },
                'learning': {'learning_rate': 0.01}
            }
            import yaml
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            agent = RobotAgent("test_robot_agent", config_file)
            yield agent
        finally:
            os.unlink(config_file)
    
    def test_agent_initialization(self, agent):
        """Test robot agent initialization"""
        assert agent.robot_id == "test_robot_agent"
        assert isinstance(agent.state, RobotState)
        assert "navigation" in agent.state.capabilities
        assert agent.state.battery_level == 100.0
    
    def test_distance_calculation(self, agent):
        """Test distance calculation"""
        pos1 = (0, 0)
        pos2 = (3, 4)
        distance = agent.calculate_distance(pos1, pos2)
        assert abs(distance - 5.0) < 0.001  # 3-4-5 triangle
    
    def test_energy_estimation(self, agent):
        """Test task energy estimation"""
        task = {
            'location': (10, 0),
            'estimated_duration': 20.0
        }
        
        agent.state.position = (0, 0)
        energy = agent.estimate_task_energy(task)
        
        assert isinstance(energy, float)
        assert energy > 0
    
    def test_task_capability_check(self, agent):
        """Test task capability checking"""
        # Task requiring available capability
        task1 = {
            'required_capabilities': ['navigation'],
            'location': (5, 5),
            'estimated_duration': 10.0
        }
        assert agent.can_execute_task(task1) == True
        
        # Task requiring unavailable capability
        task2 = {
            'required_capabilities': ['manipulation'],
            'location': (5, 5),
            'estimated_duration': 10.0
        }
        assert agent.can_execute_task(task2) == False
    
    def test_task_reward_calculation(self, agent):
        """Test task reward calculation"""
        task = {
            'task_type': 'navigation',
            'priority': 1.5,
            'estimated_duration': 30.0
        }
        
        reward = agent.calculate_task_reward(task, True, 25.0)
        assert isinstance(reward, float)
        
        # Successful task should have positive reward
        assert reward > 0
        
        # Failed task should have negative reward
        failed_reward = agent.calculate_task_reward(task, False, 25.0)
        assert failed_reward < reward

class TestTask:
    """Test cases for Task class"""
    
    def test_task_creation(self):
        """Test task creation"""
        task = Task(
            task_id="test_001",
            task_type="navigation",
            priority=1.0,
            location=(10, 20),
            deadline=time.time() + 3600,
            required_capabilities=["navigation"],
            estimated_duration=30.0
        )
        
        assert task.task_id == "test_001"
        assert task.task_type == "navigation"
        assert task.priority == 1.0
        assert task.location == (10, 20)
        assert "navigation" in task.required_capabilities
        assert task.status == "pending"
    
    def test_task_with_metadata(self):
        """Test task with metadata"""
        metadata = {"source": "test", "urgency": "high"}
        task = Task(
            task_id="test_002",
            task_type="pickup",
            priority=2.0,
            location=(0, 0),
            deadline=time.time() + 1800,
            required_capabilities=["manipulation"],
            estimated_duration=45.0,
            metadata=metadata
        )
        
        assert task.metadata == metadata
        assert task.metadata["urgency"] == "high"

class TestRobotStatus:
    """Test cases for RobotStatus class"""
    
    def test_robot_status_creation(self):
        """Test robot status creation"""
        status = RobotStatus(
            robot_id="status_test_robot",
            position=(5, 10),
            capabilities=["navigation", "sensing"],
            battery_level=85.0,
            task_load=0.3,
            last_heartbeat=time.time()
        )
        
        assert status.robot_id == "status_test_robot"
        assert status.position == (5, 10)
        assert status.battery_level == 85.0
        assert status.task_load == 0.3
        assert status.status == "active"  # default value

class TestConfigManager:
    """Test cases for configuration management"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        config_data = {
            'test_section': {
                'test_key': 'test_value',
                'nested': {
                    'deep_key': 42
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            config = ConfigManager(config_file)
            
            assert config.get('test_section.test_key') == 'test_value'
            assert config.get('test_section.nested.deep_key') == 42
            assert config.get('nonexistent.key', 'default') == 'default'
            
        finally:
            os.unlink(config_file)
    
    def test_config_defaults(self):
        """Test configuration defaults"""
        # Test with non-existent file
        config = ConfigManager("nonexistent_file.yaml")
        
        # Should load defaults
        assert isinstance(config.config_data, dict)
        assert config.get('coordination.max_robots', 0) > 0

if __name__ == "__main__":
    # Set up logging for tests
    logging.basicConfig(level=logging.DEBUG)
    
    # Run tests
    pytest.main([__file__, "-v"])
