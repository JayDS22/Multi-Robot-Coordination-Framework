#!/usr/bin/env python3
"""
ROS Interface for Multi-Robot Coordination Framework
Handles ROS2 communication between robots and coordination master
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict, deque

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from std_msgs.msg import String, Header
    from geometry_msgs.msg import Point, PoseStamped
    from nav_msgs.msg import OccupancyGrid
    from sensor_msgs.msg import BatteryState
    ROS_AVAILABLE = True
except ImportError:
    # Fallback for environments without ROS2
    ROS_AVAILABLE = False
    Node = object
    logging.warning("ROS2 not available, using simulation mode")

@dataclass
class Message:
    """Generic message structure for communication"""
    sender_id: str
    receiver_id: str
    message_type: str
    timestamp: float
    data: Dict
    message_id: str = ""
    priority: int = 1

class ROSInterface:
    """ROS2 communication interface for multi-robot coordination"""
    
    def __init__(self, node_name: str = "coordination_node"):
        self.node_name = node_name
        self.node = None
        self.initialized = False
        
        # Message handling
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_messages = deque()
        self.message_history = deque(maxlen=1000)
        
        # Topic names
        self.topics = {
            'heartbeat': '/multi_robot/heartbeat',
            'task_assignment': '/multi_robot/task_assignment',
            'task_completion': '/multi_robot/task_completion',
            'coordination': '/multi_robot/coordination',
            'emergency': '/multi_robot/emergency',
            'status': '/multi_robot/status'
        }
        
        # Publishers and subscribers
        self.publishers: Dict[str, Any] = {}
        self.subscribers: Dict[str, Any] = {}
        
        # Communication metrics
        self.messages_sent = 0
        self.messages_received = 0
        self.communication_latency = deque(maxlen=100)
        self.failed_transmissions = 0
        
        # Quality of Service profiles
        self.qos_profiles = {
            'reliable': QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            ) if ROS_AVAILABLE else None,
            'best_effort': QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            ) if ROS_AVAILABLE else None
        }
        
        self.logger = logging.getLogger(f"ros_interface_{node_name}")
        
        # Start ROS spinning in background thread
        self.ros_thread = None
        self.executor = None
        
    async def initialize(self):
        """Initialize ROS2 interface"""
        if not ROS_AVAILABLE:
            self.logger.warning("ROS2 not available, using simulation mode")
            self.initialized = True
            return
        
        try:
            # Initialize ROS2
            if not rclpy.ok():
                rclpy.init()
            
            # Create node
            self.node = CoordinationNode(self.node_name, self)
            
            # Setup publishers
            await self._setup_publishers()
            
            # Setup subscribers
            await self._setup_subscribers()
            
            # Start ROS spinning thread
            self._start_ros_thread()
            
            self.initialized = True
            self.logger.info(f"ROS interface initialized for {self.node_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ROS interface: {e}")
            # Fall back to simulation mode
            self.initialized = True
    
    async def _setup_publishers(self):
        """Setup ROS2 publishers"""
        if not self.node:
            return
        
        # Create publishers for each topic
        for topic_name, topic_path in self.topics.items():
            if topic_name in ['emergency']:
                qos = self.qos_profiles['reliable']
            else:
                qos = self.qos_profiles['best_effort']
            
            publisher = self.node.create_publisher(String, topic_path, qos)
            self.publishers[topic_name] = publisher
            
            self.logger.debug(f"Created publisher for {topic_path}")
    
    async def _setup_subscribers(self):
        """Setup ROS2 subscribers"""
        if not self.node:
            return
        
        # Create subscribers for each topic
        for topic_name, topic_path in self.topics.items():
            if topic_name in ['emergency']:
                qos = self.qos_profiles['reliable']
            else:
                qos = self.qos_profiles['best_effort']
            
            callback = lambda msg, tn=topic_name: self._message_callback(msg, tn)
            subscriber = self.node.create_subscription(String, topic_path, callback, qos)
            self.subscribers[topic_name] = subscriber
            
            self.logger.debug(f"Created subscriber for {topic_path}")
    
    def _start_ros_thread(self):
        """Start ROS spinning in background thread"""
        if not self.node:
            return
        
        def spin_node():
            try:
                rclpy.spin(self.node)
            except Exception as e:
                self.logger.error(f"ROS spinning error: {e}")
        
        self.ros_thread = threading.Thread(target=spin_node, daemon=True)
        self.ros_thread.start()
        
        self.logger.info("ROS spinning thread started")
    
    def _message_callback(self, msg, topic_name: str):
        """Handle incoming ROS messages"""
        try:
            # Parse message
            message_data = json.loads(msg.data)
            
            # Create message object
            message = Message(
                sender_id=message_data.get('sender_id', 'unknown'),
                receiver_id=message_data.get('receiver_id', 'all'),
                message_type=message_data.get('message_type', topic_name),
                timestamp=message_data.get('timestamp', time.time()),
                data=message_data.get('data', {}),
                message_id=message_data.get('message_id', ''),
                priority=message_data.get('priority', 1)
            )
            
            # Calculate latency
            current_time = time.time()
            latency = (current_time - message.timestamp) * 1000  # Convert to ms
            self.communication_latency.append(latency)
            
            # Store in pending messages
            self.pending_messages.append(message)
            self.message_history.append(message)
            self.messages_received += 1
            
            # Call registered handler if available
            if topic_name in self.message_handlers:
                try:
                    self.message_handlers[topic_name](message)
                except Exception as e:
                    self.logger.error(f"Message handler error for {topic_name}: {e}")
            
            self.logger.debug(f"Received message: {message.message_type} from {message.sender_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to process message: {e}")
    
    async def publish_message(self, topic_name: str, message: Message) -> bool:
        """Publish message to ROS topic"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Prepare message data
            message_data = {
                'sender_id': message.sender_id,
                'receiver_id': message.receiver_id,
                'message_type': message.message_type,
                'timestamp': message.timestamp,
                'data': message.data,
                'message_id': message.message_id,
                'priority': message.priority
            }
            
            if ROS_AVAILABLE and topic_name in self.publishers:
                # Publish via ROS2
                ros_msg = String()
                ros_msg.data = json.dumps(message_data)
                self.publishers[topic_name].publish(ros_msg)
            else:
                # Simulation mode - just log
                self.logger.debug(f"[SIM] Publishing to {topic_name}: {message.message_type}")
            
            self.messages_sent += 1
            self.logger.debug(f"Published message: {message.message_type} to {topic_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish message: {e}")
            self.failed_transmissions += 1
            return False
    
    async def send_heartbeat(self, heartbeat_data: Dict) -> bool:
        """Send heartbeat message"""
        message = Message(
            sender_id=heartbeat_data['robot_id'],
            receiver_id='coordination_master',
            message_type='heartbeat',
            timestamp=time.time(),
            data=heartbeat_data,
            message_id=f"hb_{heartbeat_data['robot_id']}_{int(time.time())}"
        )
        
        return await self.publish_message('heartbeat', message)
    
    async def send_task_assignment(self, robot_id: str, task_data: Dict) -> bool:
        """Send task assignment to robot"""
        message = Message(
            sender_id='coordination_master',
            receiver_id=robot_id,
            message_type='task_assignment',
            timestamp=time.time(),
            data=task_data,
            message_id=f"task_{task_data.get('task_id', 'unknown')}_{int(time.time())}"
        )
        
        return await self.publish_message('task_assignment', message)
    
    async def report_task_completion(self, completion_data: Dict) -> bool:
        """Report task completion"""
        message = Message(
            sender_id=completion_data['robot_id'],
            receiver_id='coordination_master',
            message_type='task_completion',
            timestamp=time.time(),
            data=completion_data,
            message_id=f"comp_{completion_data['task_id']}_{int(time.time())}"
        )
        
        return await self.publish_message('task_completion', message)
    
    async def register_with_master(self, registration_data: Dict) -> bool:
        """Register robot with coordination master"""
        message = Message(
            sender_id=registration_data['robot_id'],
            receiver_id='coordination_master',
            message_type='registration',
            timestamp=time.time(),
            data=registration_data,
            message_id=f"reg_{registration_data['robot_id']}_{int(time.time())}"
        )
        
        return await self.publish_message('coordination', message)
    
    async def confirm_task_assignment(self, task_id: str) -> bool:
        """Confirm task assignment acceptance"""
        message = Message(
            sender_id=self.node_name,
            receiver_id='coordination_master',
            message_type='task_confirmation',
            timestamp=time.time(),
            data={'task_id': task_id, 'status': 'accepted'},
            message_id=f"conf_{task_id}_{int(time.time())}"
        )
        
        return await self.publish_message('coordination', message)
    
    async def reject_task_assignment(self, rejection_data: Dict) -> bool:
        """Reject task assignment"""
        message = Message(
            sender_id=rejection_data['robot_id'],
            receiver_id='coordination_master',
            message_type='task_rejection',
            timestamp=time.time(),
            data=rejection_data,
            message_id=f"rej_{rejection_data['task_id']}_{int(time.time())}"
        )
        
        return await self.publish_message('coordination', message)
    
    async def receive_task_assignment(self) -> Optional[Dict]:
        """Receive task assignment (non-blocking)"""
        # Check pending messages for task assignments
        for i, message in enumerate(self.pending_messages):
            if (message.message_type == 'task_assignment' and 
                message.receiver_id in [self.node_name, 'all']):
                # Remove from pending and return
                del self.pending_messages[i]
                return message.data
        
        return None
    
    async def receive_heartbeat(self) -> Optional[Dict]:
        """Receive heartbeat message (non-blocking)"""
        for i, message in enumerate(self.pending_messages):
            if message.message_type == 'heartbeat':
                del self.pending_messages[i]
                return message.data
        
        return None
    
    async def receive_task_completion(self) -> Optional[Dict]:
        """Receive task completion report (non-blocking)"""
        for i, message in enumerate(self.pending_messages):
            if message.message_type == 'task_completion':
                del self.pending_messages[i]
                return message.data
        
        return None
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register message handler for specific message type"""
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered handler for {message_type}")
    
    def unregister_message_handler(self, message_type: str):
        """Unregister message handler"""
        if message_type in self.message_handlers:
            del self.message_handlers[message_type]
            self.logger.info(f"Unregistered handler for {message_type}")
    
    async def broadcast_emergency(self, emergency_data: Dict) -> bool:
        """Broadcast emergency message"""
        message = Message(
            sender_id=emergency_data.get('robot_id', self.node_name),
            receiver_id='all',
            message_type='emergency',
            timestamp=time.time(),
            data=emergency_data,
            message_id=f"emerg_{int(time.time())}",
            priority=0  # Highest priority
        )
        
        return await self.publish_message('emergency', message)
    
    async def send_status_update(self, status_data: Dict) -> bool:
        """Send status update"""
        message = Message(
            sender_id=status_data.get('robot_id', self.node_name),
            receiver_id='coordination_master',
            message_type='status_update',
            timestamp=time.time(),
            data=status_data,
            message_id=f"status_{int(time.time())}"
        )
        
        return await self.publish_message('status', message)
    
    def get_communication_metrics(self) -> Dict:
        """Get communication performance metrics"""
        avg_latency = sum(self.communication_latency) / len(self.communication_latency) if self.communication_latency else 0.0
        min_latency = min(self.communication_latency) if self.communication_latency else 0.0
        max_latency = max(self.communication_latency) if self.communication_latency else 0.0
        
        total_messages = self.messages_sent + self.messages_received
        success_rate = (total_messages - self.failed_transmissions) / max(total_messages, 1)
        
        return {
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'failed_transmissions': self.failed_transmissions,
            'avg_latency_ms': avg_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'success_rate': success_rate,
            'pending_messages': len(self.pending_messages),
            'total_latency_samples': len(self.communication_latency)
        }
    
    async def test_communication(self, target_node: str = 'all') -> Dict:
        """Test communication with target node"""
        test_start = time.time()
        
        # Send test message
        test_message = Message(
            sender_id=self.node_name,
            receiver_id=target_node,
            message_type='communication_test',
            timestamp=test_start,
            data={'test_id': f"test_{int(test_start)}", 'payload_size': 1024},
            message_id=f"test_{int(test_start)}"
        )
        
        success = await self.publish_message('coordination', test_message)
        
        # Wait for response (simplified)
        await asyncio.sleep(0.1)
        
        test_duration = (time.time() - test_start) * 1000  # ms
        
        return {
            'success': success,
            'duration_ms': test_duration,
            'target_node': target_node,
            'timestamp': test_start
        }
    
    def cleanup(self):
        """Cleanup ROS interface"""
        try:
            if self.node and ROS_AVAILABLE:
                self.node.destroy_node()
            
            if rclpy.ok() and ROS_AVAILABLE:
                rclpy.shutdown()
            
            self.logger.info("ROS interface cleaned up")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


if ROS_AVAILABLE:
    class CoordinationNode(Node):
        """ROS2 node for coordination communication"""
        
        def __init__(self, node_name: str, interface):
            super().__init__(node_name)
            self.interface = interface
            self.logger = logging.getLogger(f"coordination_node_{node_name}")
            
            # Node-specific parameters
            self.declare_parameter('update_rate', 10.0)
            self.declare_parameter('heartbeat_timeout', 5.0)
            self.declare_parameter('max_retries', 3)
            
            # Create timer for periodic tasks
            self.timer = self.create_timer(
                1.0 / self.get_parameter('update_rate').value,
                self.timer_callback
            )
            
            self.logger.info(f"Coordination node {node_name} created")
        
        def timer_callback(self):
            """Periodic timer callback"""
            # Handle periodic tasks like status updates, health checks, etc.
            pass
        
        def get_node_info(self) -> Dict:
            """Get node information"""
            return {
                'node_name': self.get_name(),
                'namespace': self.get_namespace(),
                'update_rate': self.get_parameter('update_rate').value,
                'heartbeat_timeout': self.get_parameter('heartbeat_timeout').value
            }


class MessageQueue:
    """Thread-safe message queue for inter-node communication"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
        # Priority queues
        self.priority_queues = {
            0: deque(),  # Emergency
            1: deque(),  # High
            2: deque(),  # Normal
            3: deque()   # Low
        }
    
    def put(self, message: Message, timeout: Optional[float] = None) -> bool:
        """Add message to queue"""
        with self.condition:
            if len(self.queue) >= self.max_size:
                if timeout:
                    self.condition.wait(timeout)
                    if len(self.queue) >= self.max_size:
                        return False
                else:
                    # Remove oldest message
                    self.queue.popleft()
            
            # Add to appropriate priority queue
            priority = min(max(message.priority, 0), 3)
            self.priority_queues[priority].append(message)
            self.queue.append(message)
            
            self.condition.notify()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Get message from queue (priority order)"""
        with self.condition:
            # Check priority queues in order
            for priority in sorted(self.priority_queues.keys()):
                if self.priority_queues[priority]:
                    message = self.priority_queues[priority].popleft()
                    # Also remove from main queue
                    try:
                        self.queue.remove(message)
                    except ValueError:
                        pass
                    return message
            
            # Wait for new message
            if timeout:
                self.condition.wait(timeout)
                # Try again after wait
                for priority in sorted(self.priority_queues.keys()):
                    if self.priority_queues[priority]:
                        message = self.priority_queues[priority].popleft()
                        try:
                            self.queue.remove(message)
                        except ValueError:
                            pass
                        return message
            
            return None
    
    def size(self) -> int:
        """Get queue size"""
        with self.lock:
            return len(self.queue)
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        with self.lock:
            return len(self.queue) == 0


async def main():
    """Test the ROS interface"""
    print("Testing ROS Interface")
    
    # Create interface
    interface = ROSInterface("test_node")
    await interface.initialize()
    
    # Test message sending
    heartbeat_data = {
        'robot_id': 'test_robot',
        'position': (1.0, 2.0),
        'battery_level': 85.0,
        'status': 'active'
    }
    
    success = await interface.send_heartbeat(heartbeat_data)
    print(f"Heartbeat sent: {success}")
    
    # Test communication metrics
    metrics = interface.get_communication_metrics()
    print(f"Communication metrics: {metrics}")
    
    # Test communication
    test_result = await interface.test_communication()
    print(f"Communication test: {test_result}")
    
    # Cleanup
    interface.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
