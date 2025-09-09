# Multi-Robot Coordination Framework API Documentation

## Overview

The Multi-Robot Coordination Framework provides a comprehensive API for managing distributed multi-agent systems. This documentation covers all major APIs, interfaces, and integration points.

## Table of Contents

1. [Core API](#core-api)
2. [Robot Agent API](#robot-agent-api)
3. [Task Management API](#task-management-api)
4. [Learning Algorithms API](#learning-algorithms-api)
5. [Communication API](#communication-api)
6. [Monitoring API](#monitoring-api)
7. [Configuration API](#configuration-api)
8. [ROS2 Integration](#ros2-integration)

---

## Core API

### CoordinationMaster

The central coordination node that manages the entire multi-robot system.

#### Class: `CoordinationMaster`

```python
class CoordinationMaster:
    def __init__(self, config_file: str = "config/system_config.yaml")
    
    async def start(self) -> None
    async def register_robot(self, robot_id: str, capabilities: List[str], 
                           position: Tuple[float, float] = (0.0, 0.0)) -> bool
    async def submit_task(self, task: Task) -> None
    async def handle_task_completion(self, robot_id: str, task_id: str, success: bool) -> None
    
    def get_system_status(self) -> Dict
    def calculate_performance_metrics(self) -> Dict
```

#### Methods

**`register_robot(robot_id, capabilities, position)`**
- **Purpose**: Register a new robot with the coordination system
- **Parameters**:
  - `robot_id` (str): Unique identifier for the robot
  - `capabilities` (List[str]): List of robot capabilities
  - `position` (Tuple[float, float]): Initial robot position
- **Returns**: `bool` - Success status
- **Example**:
```python
success = await master.register_robot("robot_1", ["navigation", "manipulation"], (0, 0))
```

**`submit_task(task)`**
- **Purpose**: Submit a new task for allocation to robots
- **Parameters**:
  - `task` (Task): Task object containing all task details
- **Returns**: None
- **Example**:
```python
task = Task(
    task_id="task_001",
    task_type="pickup",
    priority=1.5,
    location=(10, 20),
    deadline=time.time() + 3600,
    required_capabilities=["navigation", "manipulation"],
    estimated_duration=30.0
)
await master.submit_task(task)
```

**`get_system_status()`**
- **Purpose**: Get current system status and metrics
- **Returns**: `Dict` containing system information
- **Example**:
```python
status = master.get_system_status()
print(f"Active robots: {status['active_robots']}")
print(f"Pending tasks: {status['pending_tasks']}")
```

---

## Robot Agent API

### RobotAgent

Individual robot agent with autonomous decision-making capabilities.

#### Class: `RobotAgent`

```python
class RobotAgent:
    def __init__(self, robot_id: str = None, config_file: str = "config/robot_config.yaml")
    
    async def start(self) -> None
    async def execute_task(self, task: Dict) -> bool
    
    def can_execute_task(self, task: Dict) -> bool
    def get_performance_metrics(self) -> Dict
```

#### Methods

**`execute_task(task)`**
- **Purpose**: Execute an assigned task
- **Parameters**:
  - `task` (Dict): Task details from coordination master
- **Returns**: `bool` - Task execution success
- **Example**:
```python
task_data = {
    'task_id': 'task_001',
    'task_type': 'navigation',
    'location': (10, 15),
    'deadline': time.time() + 1800
}
success = await agent.execute_task(task_data)
```

**`can_execute_task(task)`**
- **Purpose**: Check if robot can execute a given task
- **Parameters**:
  - `task` (Dict): Task to evaluate
- **Returns**: `bool` - Capability to execute
- **Example**:
```python
if agent.can_execute_task(task_data):
    print("Robot can execute this task")
```

---

## Task Management API

### Task

Task representation and management.

#### Class: `Task`

```python
@dataclass
class Task:
    task_id: str
    task_type: str
    priority: float
    location: Tuple[float, float]
    deadline: float
    required_capabilities: List[str]
    estimated_duration: float
    reward: float = 0.0
    assigned_robot: Optional[str] = None
    status: str = "pending"
```

#### Task Types

- **navigation**: Movement tasks
- **pickup**: Object collection tasks
- **delivery**: Object delivery tasks
- **inspection**: Monitoring and sensing tasks
- **maintenance**: Repair and upkeep tasks
- **cleaning**: Environment cleaning tasks
- **security**: Security patrol tasks
- **emergency**: Emergency response tasks

#### Task Status Values

- `pending`: Task submitted but not assigned
- `assigned`: Task assigned to a robot
- `in_progress`: Task being executed
- `completed`: Task successfully completed
- `failed`: Task execution failed

### TaskGenerator

Generates realistic tasks for testing and operation.

#### Class: `TaskGenerator`

```python
class TaskGenerator:
    def __init__(self, config_file: str = "config/system_config.yaml")
    
    async def start_generation(self) -> None
    def generate_task(self) -> Task
    async def generate_scenario_tasks(self, scenario: str) -> List[Task]
    
    def set_generation_rate(self, rate: float) -> None
    def set_complexity(self, complexity: str) -> None
    def get_statistics(self) -> Dict
```

#### Methods

**`generate_task()`**
- **Purpose**: Generate a single realistic task
- **Returns**: `Task` object
- **Example**:
```python
generator = TaskGenerator()
task = generator.generate_task()
print(f"Generated task: {task.task_id} - {task.task_type}")
```

**`generate_scenario_tasks(scenario)`**
- **Purpose**: Generate tasks for specific test scenarios
- **Parameters**:
  - `scenario` (str): Scenario name ("emergency", "maintenance_day", etc.)
- **Returns**: `List[Task]`
- **Example**:
```python
emergency_tasks = await generator.generate_scenario_tasks("emergency")
```

---

## Learning Algorithms API

### Q-Learning

#### Class: `QLearningAgent`

```python
class QLearningAgent:
    def __init__(self, agent_id: str, learning_rate: float = 0.01, 
                 discount_factor: float = 0.95, exploration_rate: float = 0.15)
    
    def select_action(self, state: Tuple, available_actions: List[str] = None) -> str
    def update_q_value(self, state: Tuple, action: str, reward: float, 
                      next_state: Optional[Tuple] = None) -> None
    def calculate_convergence(self) -> float
```

#### Class: `QLearningCoordinator`

```python
class QLearningCoordinator:
    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.95)
    
    def add_agent(self, agent_id: str) -> QLearningAgent
    def calculate_convergence(self) -> float
    def calculate_policy_gradient(self) -> float
    def get_system_performance(self) -> Dict
```

#### Methods

**`update_q_value(state, action, reward, next_state)`**
- **Purpose**: Update Q-value based on experience
- **Parameters**:
  - `state` (Tuple): Current state representation
  - `action` (str): Action taken
  - `reward` (float): Reward received
  - `next_state` (Optional[Tuple]): Next state (if available)
- **Example**:
```python
agent.update_q_value(
    state=("navigation", 80, 0.2, 10, 15),
    action="execute",
    reward=1.5
)
```

### Auction Algorithm

#### Class: `AuctionAllocator`

```python
class AuctionAllocator:
    def __init__(self, auction_timeout: float = 5.0, min_bid_threshold: float = 0.1)
    
    async def run_auction(self, task: Task, available_robots: List[str], 
                         robot_states: Dict) -> Tuple[Optional[str], float]
    def get_auction_statistics(self) -> Dict
```

#### Methods

**`run_auction(task, available_robots, robot_states)`**
- **Purpose**: Run auction for task allocation
- **Parameters**:
  - `task` (Task): Task to allocate
  - `available_robots` (List[str]): List of available robot IDs
  - `robot_states` (Dict): Current robot state information
- **Returns**: `Tuple[Optional[str], float]` - (winning_robot, winning_bid)
- **Example**:
```python
allocator = AuctionAllocator()
winner, bid = await allocator.run_auction(task, robot_list, robot_states)
if winner:
    print(f"Task allocated to {winner} with bid {bid:.3f}")
```

---

## Communication API

### ROS Interface

#### Class: `ROSInterface`

```python
class ROSInterface:
    def __init__(self, node_name: str = "coordination_node")
    
    async def initialize(self) -> None
    async def send_heartbeat(self, heartbeat_data: Dict) -> bool
    async def send_task_assignment(self, robot_id: str, task_data: Dict) -> bool
    async def report_task_completion(self, completion_data: Dict) -> bool
    
    def get_communication_metrics(self) -> Dict
```

#### Methods

**`send_task_assignment(robot_id, task_data)`**
- **Purpose**: Send task assignment to specific robot
- **Parameters**:
  - `robot_id` (str): Target robot identifier
  - `task_data` (Dict): Task information
- **Returns**: `bool` - Send success status
- **Example**:
```python
ros_interface = ROSInterface()
await ros_interface.initialize()

task_data = {
    'task_id': 'task_001',
    'task_type': 'pickup',
    'location': (10, 20)
}
success = await ros_interface.send_task_assignment('robot_1', task_data)
```

### Message Broker

#### Class: `MessageBroker`

```python
class MessageBroker:
    def __init__(self)
    
    async def send_message(self, message: Message) -> bool
    def register_endpoint(self, robot_id: str, endpoint: str) -> None
    def subscribe(self, message_type: str, callback: Callable) -> None
    
    def get_statistics(self) -> Dict
```

### Fault Tolerance

#### Class: `FaultToleranceManager`

```python
class FaultToleranceManager:
    def __init__(self)
    
    async def report_fault(self, fault_type: FaultType, severity: FaultSeverity, 
                          robot_id: str, description: str) -> str
    async def handle_robot_failure(self, robot_id: str, tasks: Dict) -> List[str]
    
    def get_fault_statistics(self) -> Dict
```

---

## Monitoring API

### System Monitor

#### Class: `SystemMonitor`

```python
class SystemMonitor:
    def __init__(self, config_file: str = "config/system_config.yaml")
    
    async def start_monitoring(self) -> None
    async def collect_system_status(self) -> SystemStatus
    
    def get_current_status(self) -> Dict
```

#### Methods

**`collect_system_status()`**
- **Purpose**: Collect current system status snapshot
- **Returns**: `SystemStatus` object with current metrics
- **Example**:
```python
monitor = SystemMonitor()
status = await monitor.collect_system_status()
print(f"System efficiency: {status.system_efficiency:.3f}")
print(f"Active robots: {status.active_robots}/{status.total_robots}")
```

---

## Configuration API

### ConfigManager

#### Class: `ConfigManager`

```python
class ConfigManager:
    def __init__(self, config_file: str)
    
    def get(self, key: str, default: Any = None) -> Any
    def set(self, key: str, value: Any) -> None
    def save_config(self, filename: Optional[str] = None) -> None
```

#### Methods

**`get(key, default)`**
- **Purpose**: Get configuration value using dot notation
- **Parameters**:
  - `key` (str): Configuration key (supports dot notation)
  - `default` (Any): Default value if key not found
- **Returns**: Configuration value
- **Example**:
```python
config = ConfigManager("config/system_config.yaml")
max_robots = config.get("coordination.max_robots", 10)
learning_rate = config.get("learning.learning_rate", 0.01)
```

---

## ROS2 Integration

### Topics

#### Standard Topics

- `/multi_robot/heartbeat` - Robot heartbeat messages
- `/multi_robot/task_assignment` - Task assignment messages
- `/multi_robot/task_completion` - Task completion reports
- `/multi_robot/coordination` - General coordination messages
- `/multi_robot/emergency` - Emergency broadcasts
- `/multi_robot/status` - Status updates

#### Message Types

All messages use `std_msgs/String` with JSON payload for flexibility.

#### Example Message Structure

```json
{
  "sender_id": "robot_1",
  "receiver_id": "coordination_master",
  "message_type": "heartbeat",
  "timestamp": 1234567890.123,
  "data": {
    "position": [10.5, 20.3],
    "battery_level": 75.0,
    "status": "active",
    "current_task": "task_001"
  },
  "message_id": "hb_robot_1_1234567890",
  "priority": 1
}
```

### Launch Files

#### Standard Launch

```bash
ros2 launch multi_robot_coordination multi_robot.launch.py \
    num_robots:=5 \
    environment:=warehouse \
    enable_monitoring:=true
```

#### Simulation Launch

```bash
ros2 launch multi_robot_coordination simulation.launch.py \
    num_robots:=10 \
    scenario:=basic_navigation \
    gui:=true
```

### Services

#### Robot Registration Service

```bash
ros2 service call /coordination_master/register_robot \
    multi_robot_coordination_msgs/RegisterRobot \
    "robot_id: 'robot_1'
     capabilities: ['navigation', 'manipulation']
     position: {x: 0.0, y: 0.0}"
```

#### System Status Service

```bash
ros2 service call /coordination_master/get_status \
    std_srvs/Empty
```

---

## Error Handling

### Exception Types

```python
class CoordinationError(Exception):
    """Base exception for coordination errors"""
    pass

class RobotRegistrationError(CoordinationError):
    """Robot registration failed"""
    pass

class TaskAllocationError(CoordinationError):
    """Task allocation failed"""
    pass

class CommunicationError(CoordinationError):
    """Communication failure"""
    pass
```

### Error Codes

- `1001`: Robot registration failed
- `1002`: Task allocation timeout
- `1003`: Communication failure
- `1004`: Configuration error
- `1005`: Learning algorithm error
- `2001`: Robot hardware fault
- `2002`: Battery critical
- `2003`: Navigation error
- `3001`: System overload
- `3002`: Resource unavailable

---

## Performance Metrics

### Available Metrics

- **System Efficiency**: Overall task completion efficiency
- **Allocation Time**: Average task allocation time (ms)
- **Communication Latency**: Message round-trip time (ms)
- **Convergence Rate**: Q-learning convergence percentage
- **Policy Gradient**: Policy optimization metric
- **System Availability**: Percentage uptime
- **Robot Utilization**: Individual robot usage statistics

### Metrics Collection

```python
# Get system-wide metrics
master = CoordinationMaster()
metrics = master.calculate_performance_metrics()

print(f"Efficiency: {metrics['efficiency']:.3f}")
print(f"Avg Allocation Time: {metrics['avg_allocation_time']:.1f}ms")
print(f"Q-Learning Convergence: {metrics['q_convergence']:.3f}")

# Get robot-specific metrics
agent = RobotAgent("robot_1")
robot_metrics = agent.get_performance_metrics()

print(f"Robot {robot_metrics['robot_id']} Success Rate: {robot_metrics['success_rate']:.3f}")
```

---

## Integration Examples

### Basic Integration

```python
import asyncio
from coordination_master import CoordinationMaster
from robot_agent import RobotAgent
from task_generator import TaskGenerator

async def main():
    # Initialize components
    master = CoordinationMaster()
    agent = RobotAgent("robot_1")
    task_gen = TaskGenerator()
    
    # Start coordination
    await master.start()
    await agent.start()
    
    # Register robot
    await master.register_robot("robot_1", ["navigation"], (0, 0))
    
    # Generate and submit task
    task = task_gen.generate_task()
    await master.submit_task(task)
    
    # Monitor system
    while True:
        status = master.get_system_status()
        print(f"System status: {status}")
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
```

### ROS2 Integration

```python
import rclpy
from rclpy.node import Node
from coordination_master import CoordinationMaster

class CoordinationNode(Node):
    def __init__(self):
        super().__init__('coordination_node')
        self.master = CoordinationMaster()
        
        # Create ROS2 services and topics
        self.create_service(RegisterRobot, 'register_robot', self.register_robot_callback)
        self.create_timer(1.0, self.status_callback)
    
    def register_robot_callback(self, request, response):
        success = await self.master.register_robot(
            request.robot_id,
            request.capabilities,
            (request.position.x, request.position.y)
        )
        response.success = success
        return response

def main():
    rclpy.init()
    node = CoordinationNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

---

## Best Practices

### Configuration Management

1. **Use Environment-Specific Configs**: Separate configurations for different deployment environments
2. **Validate Configuration**: Always validate configuration values before use
3. **Hot Reloading**: Support runtime configuration updates where possible

### Error Handling

1. **Graceful Degradation**: System should continue operating with reduced functionality
2. **Retry Logic**: Implement exponential backoff for transient failures
3. **Circuit Breakers**: Prevent cascade failures in distributed components

### Performance Optimization

1. **Batch Operations**: Group related operations to reduce overhead
2. **Caching**: Cache frequently accessed data and computations
3. **Resource Management**: Monitor and limit resource usage per component

### Testing

1. **Unit Testing**: Test individual components in isolation
2. **Integration Testing**: Test component interactions
3. **Load Testing**: Verify performance under high load
4. **Fault Injection**: Test fault tolerance and recovery

---

For more detailed examples and advanced usage patterns, see the example applications in the `examples/` directory.
