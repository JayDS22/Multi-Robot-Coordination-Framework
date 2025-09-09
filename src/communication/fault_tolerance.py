#!/usr/bin/env python3
"""
Fault Tolerance Manager for Multi-Robot Coordination
Implements fault detection, recovery, and system resilience mechanisms
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json

class FaultType(Enum):
    """Types of faults in the multi-robot system"""
    COMMUNICATION_FAILURE = "communication_failure"
    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_CRASH = "software_crash"
    BATTERY_DEPLETION = "battery_depletion"
    NAVIGATION_ERROR = "navigation_error"
    TASK_TIMEOUT = "task_timeout"
    SENSOR_MALFUNCTION = "sensor_malfunction"
    COORDINATION_FAILURE = "coordination_failure"

class FaultSeverity(Enum):
    """Severity levels for faults"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

@dataclass
class Fault:
    """Fault representation"""
    fault_id: str
    fault_type: FaultType
    severity: FaultSeverity
    affected_robot: str
    timestamp: float
    description: str
    metadata: Dict = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None
    recovery_actions: List[str] = field(default_factory=list)

@dataclass
class HealthStatus:
    """Health status of a robot or system component"""
    component_id: str
    health_score: float  # 0.0 (failed) to 1.0 (perfect)
    last_update: float
    status_details: Dict = field(default_factory=dict)
    trending: str = "stable"  # improving, degrading, stable

class FaultToleranceManager:
    """Central fault tolerance and recovery manager"""
    
    def __init__(self):
        # Fault tracking
        self.active_faults: Dict[str, Fault] = {}
        self.fault_history: List[Fault] = []
        self.fault_patterns: Dict[str, List[Fault]] = defaultdict(list)
        
        # Health monitoring
        self.robot_health: Dict[str, HealthStatus] = {}
        self.system_health_score = 1.0
        self.health_history = deque(maxlen=1000)
        
        # Recovery strategies
        self.recovery_strategies: Dict[FaultType, List[str]] = {
            FaultType.COMMUNICATION_FAILURE: [
                "restart_communication_module",
                "switch_communication_channel",
                "use_backup_communication",
                "enter_autonomous_mode"
            ],
            FaultType.HARDWARE_FAILURE: [
                "isolate_faulty_component",
                "activate_backup_system",
                "request_maintenance",
                "graceful_shutdown"
            ],
            FaultType.SOFTWARE_CRASH: [
                "restart_software",
                "load_backup_configuration",
                "reset_to_safe_state",
                "emergency_shutdown"
            ],
            FaultType.BATTERY_DEPLETION: [
                "return_to_charging_station",
                "request_battery_replacement",
                "enter_power_saving_mode",
                "emergency_shutdown"
            ],
            FaultType.NAVIGATION_ERROR: [
                "recalibrate_sensors",
                "request_manual_navigation",
                "use_backup_localization",
                "return_to_safe_position"
            ],
            FaultType.TASK_TIMEOUT: [
                "abort_current_task",
                "request_task_reassignment",
                "extend_task_deadline",
                "report_task_failure"
            ]
        }
        
        # System parameters
        self.fault_detection_threshold = 0.7
        self.health_check_interval = 2.0
        self.failover_timeout = 2.0  # Target <2s failover
        self.system_availability_target = 0.999  # 99.9%
        
        # Performance tracking
        self.availability_history = deque(maxlen=1000)
        self.failover_times = deque(maxlen=100)
        self.mtbf_data = deque(maxlen=100)  # Mean Time Between Failures
        self.mttr_data = deque(maxlen=100)  # Mean Time To Recovery
        
        # Event logging
        self.event_log = deque(maxlen=10000)
        
        self.logger = logging.getLogger("fault_tolerance")
        self.logger.info("Fault tolerance manager initialized")
    
    async def monitor_system_health(self):
        """Continuously monitor system health"""
        while True:
            try:
                await self.check_all_robot_health()
                await self.detect_system_anomalies()
                await self.update_system_metrics()
                
                # Log system status
                if len(self.health_history) % 30 == 0:  # Every minute
                    self.logger.info(f"System health: {self.system_health_score:.3f}, "
                                   f"Active faults: {len(self.active_faults)}")
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
            
            await asyncio.sleep(self.health_check_interval)
    
    async def check_robot_health(self, robot_state) -> List[str]:
        """Check individual robot health and detect faults"""
        faults = []
        robot_id = robot_state.robot_id if hasattr(robot_state, 'robot_id') else "unknown"
        current_time = time.time()
        
        # Battery health check
        if hasattr(robot_state, 'battery_level'):
            if robot_state.battery_level < 10.0:
                faults.append("critical_battery")
                await self.report_fault(
                    FaultType.BATTERY_DEPLETION,
                    FaultSeverity.CRITICAL,
                    robot_id,
                    f"Battery critically low: {robot_state.battery_level}%"
                )
            elif robot_state.battery_level < 20.0:
                faults.append("low_battery")
                await self.report_fault(
                    FaultType.BATTERY_DEPLETION,
                    FaultSeverity.HIGH,
                    robot_id,
                    f"Battery low: {robot_state.battery_level}%"
                )
        
        # Communication health check
        if hasattr(robot_state, 'last_heartbeat'):
            time_since_heartbeat = current_time - robot_state.last_heartbeat
            if time_since_heartbeat > 10.0:  # 10 seconds without heartbeat
                faults.append("communication_loss")
                await self.report_fault(
                    FaultType.COMMUNICATION_FAILURE,
                    FaultSeverity.HIGH,
                    robot_id,
                    f"No heartbeat for {time_since_heartbeat:.1f}s"
                )
        
        # Task execution health check
        if hasattr(robot_state, 'current_task') and robot_state.current_task:
            # Check for task timeout (simplified)
            task_duration = current_time - getattr(robot_state, 'task_start_time', current_time)
            if task_duration > 300.0:  # 5 minutes
                faults.append("task_timeout")
                await self.report_fault(
                    FaultType.TASK_TIMEOUT,
                    FaultSeverity.MEDIUM,
                    robot_id,
                    f"Task running for {task_duration:.1f}s"
                )
        
        # Position/navigation health check
        if hasattr(robot_state, 'position') and hasattr(robot_state, 'velocity'):
            # Check for stuck robot (not moving for extended period)
            velocity_magnitude = np.sqrt(robot_state.velocity[0]**2 + robot_state.velocity[1]**2)
            if (hasattr(robot_state, 'status') and robot_state.status == "moving" and 
                velocity_magnitude < 0.1):
                faults.append("navigation_error")
        
        # Update robot health status
        health_score = self.calculate_robot_health_score(robot_state, faults)
        self.robot_health[robot_id] = HealthStatus(
            component_id=robot_id,
            health_score=health_score,
            last_update=current_time,
            status_details={
                'battery_level': getattr(robot_state, 'battery_level', 0),
                'active_faults': len(faults),
                'fault_types': faults
            }
        )
        
        return faults
    
    def calculate_robot_health_score(self, robot_state, faults: List[str]) -> float:
        """Calculate overall health score for a robot"""
        base_score = 1.0
        
        # Battery penalty
        if hasattr(robot_state, 'battery_level'):
            battery_factor = robot_state.battery_level / 100.0
            base_score *= (0.5 + 0.5 * battery_factor)  # 50-100% based on battery
        
        # Fault penalties
        fault_penalties = {
            'critical_battery': 0.8,
            'low_battery': 0.9,
            'communication_loss': 0.7,
            'task_timeout': 0.85,
            'navigation_error': 0.9
        }
        
        for fault in faults:
            if fault in fault_penalties:
                base_score *= fault_penalties[fault]
        
        return max(0.0, base_score)
    
    async def check_all_robot_health(self):
        """Check health of all robots in the system"""
        # This would be called with actual robot states
        # For now, we'll update system health based on known robots
        
        if not self.robot_health:
            self.system_health_score = 1.0
            return
        
        # Calculate system health as average of robot health
        health_scores = [robot.health_score for robot in self.robot_health.values()]
        self.system_health_score = np.mean(health_scores)
        
        # Store in history
        self.health_history.append({
            'timestamp': time.time(),
            'system_health': self.system_health_score,
            'num_robots': len(self.robot_health),
            'avg_health': self.system_health_score
        })
    
    async def check_system_health(self, robots: Dict) -> Dict:
        """Check overall system health"""
        current_time = time.time()
        health_issues = []
        critical_issues = []
        
        # Check individual robots
        for robot_id, robot in robots.items():
            faults = await self.check_robot_health(robot)
            
            if any("critical" in fault for fault in faults):
                critical_issues.append(f"Robot {robot_id}: {faults}")
            elif faults:
                health_issues.append(f"Robot {robot_id}: {faults}")
        
        # Check system-level metrics
        active_robots = sum(1 for robot in robots.values() if robot.status == "active")
        total_robots = len(robots)
        
        if total_robots > 0:
            availability_ratio = active_robots / total_robots
            if availability_ratio < 0.8:
                critical_issues.append(f"Low system availability: {availability_ratio:.2%}")
        
        # Check communication health
        avg_latency = await self.check_communication_latency()
        if avg_latency > 50.0:  # >50ms latency
            health_issues.append(f"High communication latency: {avg_latency:.1f}ms")
        
        return {
            'system_health_score': self.system_health_score,
            'active_robots': active_robots,
            'total_robots': total_robots,
            'availability_ratio': availability_ratio if total_robots > 0 else 0.0,
            'health_issues': health_issues,
            'critical_issues': critical_issues,
            'active_faults': len(self.active_faults),
            'avg_communication_latency': avg_latency
        }
    
    async def check_communication_latency(self) -> float:
        """Check average communication latency"""
        # This would integrate with the communication system
        # For simulation, return a realistic value
        return np.random.uniform(15.0, 30.0)  # 15-30ms simulated latency
    
    async def detect_system_anomalies(self):
        """Detect system-wide anomalies and patterns"""
        current_time = time.time()
        
        # Check for fault clustering (multiple faults in short time)
        recent_faults = [f for f in self.fault_history if current_time - f.timestamp < 60.0]
        
        if len(recent_faults) > 5:  # More than 5 faults in last minute
            await self.report_fault(
                FaultType.COORDINATION_FAILURE,
                FaultSeverity.HIGH,
                "system",
                f"Fault clustering detected: {len(recent_faults)} faults in 60s"
            )
        
        # Check for degrading system health trend
        if len(self.health_history) > 10:
            recent_health = [h['system_health'] for h in list(self.health_history)[-10:]]
            health_trend = np.polyfit(range(len(recent_health)), recent_health, 1)[0]
            
            if health_trend < -0.01:  # Degrading trend
                self.logger.warning(f"System health degrading: trend = {health_trend:.4f}")
    
    async def report_fault(self, fault_type: FaultType, severity: FaultSeverity, 
                          robot_id: str, description: str, metadata: Dict = None):
        """Report a new fault in the system"""
        fault_id = f"{fault_type.value}_{robot_id}_{int(time.time())}"
        
        fault = Fault(
            fault_id=fault_id,
            fault_type=fault_type,
            severity=severity,
            affected_robot=robot_id,
            timestamp=time.time(),
            description=description,
            metadata=metadata or {}
        )
        
        # Store fault
        self.active_faults[fault_id] = fault
        self.fault_history.append(fault)
        self.fault_patterns[robot_id].append(fault)
        
        # Log event
        self.event_log.append({
            'timestamp': fault.timestamp,
            'event_type': 'fault_detected',
            'fault_id': fault_id,
            'fault_type': fault_type.value,
            'severity': severity.value,
            'robot_id': robot_id,
            'description': description
        })
        
        self.logger.warning(f"Fault detected: {fault_type.value} on {robot_id} - {description}")
        
        # Trigger immediate recovery if critical
        if severity in [FaultSeverity.CRITICAL, FaultSeverity.HIGH]:
            await self.initiate_recovery(fault)
        
        return fault_id
    
    async def initiate_recovery(self, fault: Fault):
        """Initiate recovery actions for a fault"""
        recovery_start_time = time.time()
        
        self.logger.info(f"Initiating recovery for fault {fault.fault_id}")
        
        # Get recovery strategies for this fault type
        strategies = self.recovery_strategies.get(fault.fault_type, [])
        
        if not strategies:
            self.logger.warning(f"No recovery strategies defined for {fault.fault_type}")
            return
        
        # Execute recovery strategies in order
        for strategy in strategies:
            try:
                success = await self.execute_recovery_action(fault, strategy)
                
                if success:
                    # Recovery successful
                    recovery_time = time.time() - recovery_start_time
                    await self.mark_fault_resolved(fault.fault_id, strategy, recovery_time)
                    
                    # Record successful failover time
                    self.failover_times.append(recovery_time)
                    
                    self.logger.info(f"Fault {fault.fault_id} resolved using {strategy} "
                                   f"in {recovery_time:.2f}s")
                    break
                else:
                    self.logger.warning(f"Recovery strategy {strategy} failed for {fault.fault_id}")
                    
            except Exception as e:
                self.logger.error(f"Recovery strategy {strategy} error: {e}")
        
        else:
            # All recovery strategies failed
            self.logger.error(f"All recovery strategies failed for fault {fault.fault_id}")
            await self.escalate_fault(fault)
    
    async def execute_recovery_action(self, fault: Fault, strategy: str) -> bool:
        """Execute a specific recovery action"""
        robot_id = fault.affected_robot
        
        # Simulate recovery actions (in real implementation, these would trigger actual recovery)
        self.logger.debug(f"Executing recovery action: {strategy} for robot {robot_id}")
        
        # Simulate action execution time
        await asyncio.sleep(0.1)
        
        # Simulate success rate based on strategy
        success_rates = {
            "restart_communication_module": 0.8,
            "switch_communication_channel": 0.9,
            "use_backup_communication": 0.7,
            "enter_autonomous_mode": 0.95,
            "isolate_faulty_component": 0.6,
            "activate_backup_system": 0.85,
            "request_maintenance": 1.0,
            "graceful_shutdown": 1.0,
            "restart_software": 0.9,
            "load_backup_configuration": 0.8,
            "reset_to_safe_state": 0.95,
            "return_to_charging_station": 0.9,
            "enter_power_saving_mode": 0.95,
            "abort_current_task": 1.0,
            "request_task_reassignment": 0.9
        }
        
        success_rate = success_rates.get(strategy, 0.7)
        success = np.random.random() < success_rate
        
        # Record recovery action
        fault.recovery_actions.append(strategy)
        
        # Log action
        self.event_log.append({
            'timestamp': time.time(),
            'event_type': 'recovery_action',
            'fault_id': fault.fault_id,
            'action': strategy,
            'success': success,
            'robot_id': robot_id
        })
        
        return success
    
    async def mark_fault_resolved(self, fault_id: str, resolution_method: str, resolution_time: float):
        """Mark a fault as resolved"""
        if fault_id in self.active_faults:
            fault = self.active_faults[fault_id]
            fault.resolved = True
            fault.resolution_time = resolution_time
            
            # Remove from active faults
            del self.active_faults[fault_id]
            
            # Update MTTR (Mean Time To Recovery)
            self.mttr_data.append(resolution_time)
            
            # Log resolution
            self.event_log.append({
                'timestamp': time.time(),
                'event_type': 'fault_resolved',
                'fault_id': fault_id,
                'resolution_method': resolution_method,
                'resolution_time': resolution_time
            })
            
            self.logger.info(f"Fault {fault_id} resolved in {resolution_time:.2f}s")
    
    async def escalate_fault(self, fault: Fault):
        """Escalate fault when recovery fails"""
        self.logger.error(f"Escalating fault {fault.fault_id} - recovery failed")
        
        # Update fault severity
        if fault.severity != FaultSeverity.CRITICAL:
            fault.severity = FaultSeverity.CRITICAL
        
        # Trigger emergency protocols
        if fault.fault_type == FaultType.COMMUNICATION_FAILURE:
            await self.handle_communication_emergency(fault)
        elif fault.fault_type == FaultType.HARDWARE_FAILURE:
            await self.handle_hardware_emergency(fault)
        
        # Log escalation
        self.event_log.append({
            'timestamp': time.time(),
            'event_type': 'fault_escalated',
            'fault_id': fault.fault_id,
            'robot_id': fault.affected_robot
        })
    
    async def handle_communication_emergency(self, fault: Fault):
        """Handle communication emergency"""
        robot_id = fault.affected_robot
        
        # Switch to emergency communication protocol
        self.logger.warning(f"Activating emergency communication for {robot_id}")
        
        # Notify other robots about the failure
        # In real implementation, this would broadcast emergency messages
        
    async def handle_hardware_emergency(self, fault: Fault):
        """Handle hardware emergency"""
        robot_id = fault.affected_robot
        
        # Isolate faulty robot and redistribute its tasks
        self.logger.warning(f"Isolating robot {robot_id} due to hardware failure")
        
        # In real implementation, this would trigger task reallocation
    
    async def handle_robot_failure(self, robot_id: str, tasks: Dict):
        """Handle complete robot failure"""
        self.logger.error(f"Handling complete failure of robot {robot_id}")
        
        # Find tasks assigned to failed robot
        failed_robot_tasks = []
        for task_id, task in tasks.items():
            if (hasattr(task, 'assigned_robot') and 
                task.assigned_robot == robot_id and 
                task.status in ['assigned', 'in_progress']):
                failed_robot_tasks.append(task_id)
        
        # Create fault record
        await self.report_fault(
            FaultType.HARDWARE_FAILURE,
            FaultSeverity.CRITICAL,
            robot_id,
            f"Complete robot failure, {len(failed_robot_tasks)} tasks affected"
        )
        
        # Mark robot as offline in health status
        if robot_id in self.robot_health:
            self.robot_health[robot_id].health_score = 0.0
            self.robot_health[robot_id].status_details['status'] = 'failed'
        
        return failed_robot_tasks
    
    async def update_system_metrics(self):
        """Update system-wide fault tolerance metrics"""
        current_time = time.time()
        
        # Calculate system availability
        if self.robot_health:
            operational_robots = sum(1 for robot in self.robot_health.values() 
                                   if robot.health_score > 0.5)
            total_robots = len(self.robot_health)
            availability = operational_robots / total_robots if total_robots > 0 else 0.0
        else:
            availability = 1.0
        
        self.availability_history.append({
            'timestamp': current_time,
            'availability': availability,
            'active_faults': len(self.active_faults)
        })
        
        # Calculate MTBF (Mean Time Between Failures)
        if len(self.fault_history) > 1:
            fault_intervals = []
            for i in range(1, len(self.fault_history)):
                interval = self.fault_history[i].timestamp - self.fault_history[i-1].timestamp
                fault_intervals.append(interval)
            
            if fault_intervals:
                mtbf = np.mean(fault_intervals[-10:])  # Last 10 intervals
                self.mtbf_data.append(mtbf)
    
    def get_system_availability(self) -> float:
        """Get current system availability"""
        if not self.availability_history:
            return 1.0
        
        # Calculate availability over last hour
        current_time = time.time()
        recent_data = [
            entry for entry in self.availability_history 
            if current_time - entry['timestamp'] < 3600
        ]
        
        if not recent_data:
            return 1.0
        
        return np.mean([entry['availability'] for entry in recent_data])
    
    def get_fault_statistics(self) -> Dict:
        """Get comprehensive fault tolerance statistics"""
        current_time = time.time()
        
        # Basic statistics
        total_faults = len(self.fault_history)
        resolved_faults = sum(1 for fault in self.fault_history if fault.resolved)
        
        # Availability metrics
        system_availability = self.get_system_availability()
        
        # Timing metrics
        avg_failover_time = np.mean(list(self.failover_times)) if self.failover_times else 0.0
        avg_mttr = np.mean(list(self.mttr_data)) if self.mttr_data else 0.0
        avg_mtbf = np.mean(list(self.mtbf_data)) if self.mtbf_data else 0.0
        
        # Fault type distribution
        fault_type_counts = defaultdict(int)
        for fault in self.fault_history:
            fault_type_counts[fault.fault_type.value] += 1
        
        # Recent performance (last 24 hours)
        recent_faults = [
            f for f in self.fault_history 
            if current_time - f.timestamp < 86400
        ]
        
        return {
            'total_faults': total_faults,
            'resolved_faults': resolved_faults,
            'active_faults': len(self.active_faults),
            'resolution_rate': resolved_faults / max(total_faults, 1),
            'system_availability': system_availability,
            'system_health_score': self.system_health_score,
            'avg_failover_time': avg_failover_time,
            'max_failover_time': max(self.failover_times) if self.failover_times else 0.0,
            'avg_mttr': avg_mttr,
            'avg_mtbf': avg_mtbf,
            'fault_type_distribution': dict(fault_type_counts),
            'recent_faults_24h': len(recent_faults),
            'meets_availability_target': system_availability >= self.system_availability_target,
            'meets_failover_target': avg_failover_time <= self.failover_timeout
        }


class RobotFaultHandler:
    """Individual robot fault handler"""
    
    def __init__(self, robot_id: str):
        self.robot_id = robot_id
        self.local_faults: List[Fault] = []
        self.self_diagnostics = {
            'battery_health': 1.0,
            'communication_health': 1.0,
            'sensor_health': 1.0,
            'actuator_health': 1.0,
            'software_health': 1.0
        }
        
        self.logger = logging.getLogger(f"robot_fault_handler_{robot_id}")
    
    async def run_self_diagnostics(self) -> Dict[str, float]:
        """Run self-diagnostic tests"""
        # Simulate diagnostic tests
        self.self_diagnostics['battery_health'] = np.random.uniform(0.8, 1.0)
        self.self_diagnostics['communication_health'] = np.random.uniform(0.9, 1.0)
        self.self_diagnostics['sensor_health'] = np.random.uniform(0.85, 1.0)
        self.self_diagnostics['actuator_health'] = np.random.uniform(0.9, 1.0)
        self.self_diagnostics['software_health'] = np.random.uniform(0.95, 1.0)
        
        return self.self_diagnostics.copy()
    
    async def detect_local_faults(self, robot_state) -> List[str]:
        """Detect faults specific to this robot"""
        faults = []
        
        # Run diagnostics
        diagnostics = await self.run_self_diagnostics()
        
        # Check each component
        for component, health in diagnostics.items():
            if health < 0.7:
                faults.append(f"degraded_{component}")
            elif health < 0.5:
                faults.append(f"critical_{component}")
        
        return faults


async def main():
    """Test fault tolerance system"""
    print("Testing Fault Tolerance System")
    
    # Create fault tolerance manager
    ft_manager = FaultToleranceManager()
    
    # Simulate some faults
    await ft_manager.report_fault(
        FaultType.COMMUNICATION_FAILURE,
        FaultSeverity.HIGH,
        "robot_1",
        "Lost communication with master"
    )
    
    await ft_manager.report_fault(
        FaultType.BATTERY_DEPLETION,
        FaultSeverity.CRITICAL,
        "robot_2",
        "Battery level critical: 5%"
    )
    
    # Wait for recovery
    await asyncio.sleep(1.0)
    
    # Get statistics
    stats = ft_manager.get_fault_statistics()
    print(f"Fault Statistics: {stats}")
    
    # Test robot fault handler
    robot_handler = RobotFaultHandler("test_robot")
    diagnostics = await robot_handler.run_self_diagnostics()
    print(f"Self Diagnostics: {diagnostics}")


if __name__ == "__main__":
    asyncio.run(main())
