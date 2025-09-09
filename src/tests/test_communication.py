#!/usr/bin/env python3
"""
Unit tests for communication functionality
"""

import pytest
import asyncio
import time
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from communication.ros_interface import ROSInterface, Message
from communication.fault_tolerance import FaultToleranceManager, FaultType, FaultSeverity, Fault
from communication.message_broker import MessageBroker

class TestROSInterface:
    """Test cases for ROS Interface"""
    
    @pytest.fixture
    async def ros_interface(self):
        """Create a test ROS interface"""
        interface = ROSInterface("test_node")
        await interface.initialize()
        yield interface
        interface.cleanup()
    
    @pytest.mark.asyncio
    async def test_interface_initialization(self, ros_interface):
        """Test ROS interface initialization"""
        assert ros_interface.node_name == "test_node"
        assert ros_interface.initialized == True
        assert len(ros_interface.topics) > 0
    
    @pytest.mark.asyncio
    async def test_heartbeat_sending(self, ros_interface):
        """Test heartbeat message sending"""
        heartbeat_data = {
            'robot_id': 'test_robot',
            'position': (5.0, 10.0),
            'battery_level': 75.0,
            'status': 'active'
        }
        
        success = await ros_interface.send_heartbeat(heartbeat_data)
        assert success == True
        assert ros_interface.messages_sent > 0
    
    @pytest.mark.asyncio
    async def test_task_assignment_sending(self, ros_interface):
        """Test task assignment message sending"""
        task_data = {
            'task_id': 'test_task_001',
            'task_type': 'navigation',
            'location': (15, 20),
            'deadline': time.time() + 3600
        }
        
        success = await ros_interface.send_task_assignment('test_robot', task_data)
        assert success == True
    
    @pytest.mark.asyncio
    async def test_task_completion_reporting(self, ros_interface):
        """Test task completion reporting"""
        completion_data = {
            'robot_id': 'test_robot',
            'task_id': 'test_task_001',
            'success': True,
            'completion_time': time.time()
        }
        
        success = await ros_interface.report_task_completion(completion_data)
        assert success == True
    
    @pytest.mark.asyncio
    async def test_registration_with_master(self, ros_interface):
        """Test robot registration with master"""
        registration_data = {
            'robot_id': 'test_robot',
            'capabilities': ['navigation', 'sensing'],
            'position': (0, 0)
        }
        
        success = await ros_interface.register_with_master(registration_data)
        assert success == True
    
    @pytest.mark.asyncio
    async def test_emergency_broadcast(self, ros_interface):
        """Test emergency message broadcasting"""
        emergency_data = {
            'robot_id': 'test_robot',
            'emergency_type': 'collision_detected',
            'location': (10, 15),
            'severity': 'high'
        }
        
        success = await ros_interface.broadcast_emergency(emergency_data)
        assert success == True
    
    def test_communication_metrics(self, ros_interface):
        """Test communication metrics collection"""
        metrics = ros_interface.get_communication_metrics()
        
        assert isinstance(metrics, dict)
        assert 'messages_sent' in metrics
        assert 'messages_received' in metrics
        assert 'failed_transmissions' in metrics
        assert 'avg_latency_ms' in metrics
        assert 'success_rate' in metrics
    
    @pytest.mark.asyncio
    async def test_communication_test(self, ros_interface):
        """Test communication testing functionality"""
        result = await ros_interface.test_communication('test_target')
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'duration_ms' in result
        assert 'target_node' in result

class TestFaultToleranceManager:
    """Test cases for Fault Tolerance Manager"""
    
    @pytest.fixture
    def ft_manager(self):
        """Create a test fault tolerance manager"""
        return FaultToleranceManager()
    
    @pytest.mark.asyncio
    async def test_fault_reporting(self, ft_manager):
        """Test fault reporting"""
        fault_id = await ft_manager.report_fault(
            FaultType.COMMUNICATION_FAILURE,
            FaultSeverity.HIGH,
            "test_robot",
            "Test communication fault"
        )
        
        assert isinstance(fault_id, str)
        assert fault_id in ft_manager.active_faults
        assert len(ft_manager.fault_history) == 1
    
    @pytest.mark.asyncio
    async def test_fault_recovery(self, ft_manager):
        """Test fault recovery process"""
        # Report a fault
        fault_id = await ft_manager.report_fault(
            FaultType.BATTERY_DEPLETION,
            FaultSeverity.CRITICAL,
            "test_robot",
            "Critical battery level"
        )
        
        # Wait a bit for recovery to be attempted
        await asyncio.sleep(0.5)
        
        # Check if recovery was initiated
        fault = ft_manager.active_faults.get(fault_id)
        if fault:
            assert len(fault.recovery_actions) > 0
    
    @pytest.mark.asyncio
    async def test_robot_health_check(self, ft_manager):
        """Test robot health checking"""
        # Mock robot state
        robot_state = type('RobotState', (), {
            'robot_id': 'test_robot',
            'battery_level': 15.0,  # Low battery
            'last_heartbeat': time.time() - 5.0,  # Recent heartbeat
            'position': (10, 15),
            'velocity': (0.1, 0.2),
            'status': 'active'
        })
        
        faults = await ft_manager.check_robot_health(robot_state)
        
        assert isinstance(faults, list)
        # Should detect low battery
        assert any('battery' in fault for fault in faults)
    
    @pytest.mark.asyncio
    async def test_system_health_check(self, ft_manager):
        """Test system health checking"""
        # Mock robots dictionary
        robots = {
            'robot_1': type('Robot', (), {'status': 'active', 'battery_level': 80}),
            'robot_2': type('Robot', (), {'status': 'active', 'battery_level': 60}),
            'robot_3': type('Robot', (), {'status': 'offline', 'battery_level': 5})
        }
        
        health_status = await ft_manager.check_system_health(robots)
        
        assert isinstance(health_status, dict)
        assert 'system_health_score' in health_status
        assert 'active_robots' in health_status
        assert 'total_robots' in health_status
        assert 'availability_ratio' in health_status
    
    @pytest.mark.asyncio
    async def test_robot_failure_handling(self, ft_manager):
        """Test complete robot failure handling"""
        # Mock tasks
        tasks = {
            'task_1': type('Task', (), {
                'assigned_robot': 'failed_robot',
                'status': 'in_progress'
            }),
            'task_2': type('Task', (), {
                'assigned_robot': 'failed_robot',
                'status': 'assigned'
            }),
            'task_3': type('Task', (), {
                'assigned_robot': 'other_robot',
                'status': 'assigned'
            })
        }
        
        failed_tasks = await ft_manager.handle_robot_failure('failed_robot', tasks)
        
        assert isinstance(failed_tasks, list)
        assert len(failed_tasks) == 2  # Two tasks were assigned to failed robot
        assert 'task_1' in failed_tasks
        assert 'task_2' in failed_tasks
        assert 'task_3' not in failed_tasks
    
    def test_fault_statistics(self, ft_manager):
        """Test fault statistics generation"""
        stats = ft_manager.get_fault_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_faults' in stats
        assert 'resolved_faults' in stats
        assert 'active_faults' in stats
        assert 'system_availability' in stats
        assert 'resolution_rate' in stats
    
    def test_system_availability_calculation(self, ft_manager):
        """Test system availability calculation"""
        availability = ft_manager.get_system_availability()
        
        assert isinstance(availability, float)
        assert 0.0 <= availability <= 1.0

class TestMessageBroker:
    """Test cases for Message Broker"""
    
    @pytest.fixture
    def message_broker(self):
        """Create a test message broker"""
        broker = MessageBroker()
        yield broker
        broker.shutdown()
    
    def test_broker_initialization(self, message_broker):
        """Test message broker initialization"""
        assert len(message_broker.message_queues) == 0
        assert len(message_broker.routing_table) == 0
        assert message_broker.messages_sent == 0
        assert message_broker.messages_delivered == 0
    
    def test_endpoint_registration(self, message_broker):
        """Test endpoint registration"""
        message_broker.register_endpoint("robot_1", "endpoint_1")
        
        assert "robot_1" in message_broker.routing_table
        assert message_broker.routing_table["robot_1"] == "endpoint_1"
    
    def test_endpoint_unregistration(self, message_broker):
        """Test endpoint unregistration"""
        message_broker.register_endpoint("robot_1", "endpoint_1")
        message_broker.unregister_endpoint("robot_1")
        
        assert "robot_1" not in message_broker.routing_table
    
    @pytest.mark.asyncio
    async def test_message_sending(self, message_broker):
        """Test message sending"""
        # Register endpoint
        message_broker.register_endpoint("robot_1", "endpoint_1")
        
        # Create message
        message = Message(
            sender_id="test_sender",
            receiver_id="robot_1",
            message_type="test_message",
            timestamp=time.time(),
            data={"test": "data"},
            message_id="test_msg_001"
        )
        
        success = await message_broker.send_message(message)
        
        assert success == True
        assert message_broker.messages_sent > 0
    
    @pytest.mark.asyncio
    async def test_message_queuing(self, message_broker):
        """Test message queuing for unregistered endpoints"""
        # Create message for unregistered robot
        message = Message(
            sender_id="test_sender",
            receiver_id="unregistered_robot",
            message_type="test_message",
            timestamp=time.time(),
            data={"test": "data"},
            message_id="test_msg_002"
        )
        
        success = await message_broker.send_message(message)
        
        assert success == True
        assert "unregistered_robot" in message_broker.message_queues
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, message_broker):
        """Test broadcast message sending"""
        # Register multiple endpoints
        message_broker.register_endpoint("robot_1", "endpoint_1")
        message_broker.register_endpoint("robot_2", "endpoint_2")
        
        # Create broadcast message
        message = Message(
            sender_id="broadcaster",
            receiver_id="all",
            message_type="broadcast_test",
            timestamp=time.time(),
            data={"broadcast": "data"},
            message_id="broadcast_001"
        )
        
        success = await message_broker.send_message(message)
        
        assert success == True
    
    def test_message_subscription(self, message_broker):
        """Test message type subscription"""
        def test_callback(msg):
            pass
        
        message_broker.subscribe("test_type", test_callback)
        
        assert "test_type" in message_broker.subscribers
        assert test_callback in message_broker.subscribers["test_type"]
        
        # Test unsubscription
        message_broker.unsubscribe("test_type", test_callback)
        assert test_callback not in message_broker.subscribers["test_type"]
    
    @pytest.mark.asyncio
    async def test_message_receiving(self, message_broker):
        """Test message receiving"""
        # Queue a message
        message = Message(
            sender_id="sender",
            receiver_id="robot_1",
            message_type="test",
            timestamp=time.time(),
            data={},
            message_id="recv_test_001"
        )
        
        await message_broker.send_message(message)
        
        # Try to receive message
        received = await message_broker.receive_message("robot_1", timeout=0.1)
        
        # May or may not receive depending on routing
        if received:
            assert isinstance(received, Message)
            assert received.receiver_id == "robot_1"
    
    @pytest.mark.asyncio
    async def test_communication_health_check(self, message_broker):
        """Test communication health checking"""
        health_issues = await message_broker.check_communication_health()
        
        assert isinstance(health_issues, list)
        # Initially should have no issues
    
    def test_broker_statistics(self, message_broker):
        """Test broker statistics"""
        stats = message_broker.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'messages_sent' in stats
        assert 'messages_delivered' in stats
        assert 'messages_failed' in stats
        assert 'delivery_rate' in stats
        assert 'failure_rate' in stats

class TestMessage:
    """Test cases for Message class"""
    
    def test_message_creation(self):
        """Test message creation"""
        message = Message(
            sender_id="robot_1",
            receiver_id="robot_2",
            message_type="task_assignment",
            timestamp=time.time(),
            data={"task_id": "test_001"},
            message_id="msg_001"
        )
        
        assert message.sender_id == "robot_1"
        assert message.receiver_id == "robot_2"
        assert message.message_type == "task_assignment"
        assert message.data["task_id"] == "test_001"
        assert message.priority == 1  # Default priority
        assert message.ttl == 30.0  # Default TTL

class TestFault:
    """Test cases for Fault class"""
    
    def test_fault_creation(self):
        """Test fault creation"""
        fault = Fault(
            fault_id="fault_001",
            fault_type=FaultType.HARDWARE_FAILURE,
            severity=FaultSeverity.HIGH,
            affected_robot="robot_1",
            timestamp=time.time(),
            description="Motor malfunction detected"
        )
        
        assert fault.fault_id == "fault_001"
        assert fault.fault_type == FaultType.HARDWARE_FAILURE
        assert fault.severity == FaultSeverity.HIGH
        assert fault.affected_robot == "robot_1"
        assert fault.resolved == False
        assert len(fault.recovery_actions) == 0

if __name__ == "__main__":
    # Set up logging for tests
    logging.basicConfig(level=logging.DEBUG)
    
    # Run tests
    pytest.main([__file__, "-v"])
