#!/usr/bin/env python3
"""
System Monitor for Multi-Robot Coordination Framework
Real-time monitoring, visualization, and performance analysis
"""

import asyncio
import logging
import time
import json
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import threading
import tkinter as tk
from tkinter import ttk
import queue

# Import framework components
from utils.config import ConfigManager
from utils.logger import setup_logger
from utils.metrics import MetricsCollector
from communication.ros_interface import ROSInterface

@dataclass
class SystemStatus:
    """System status snapshot"""
    timestamp: float
    active_robots: int
    total_robots: int
    pending_tasks: int
    completed_tasks: int
    failed_tasks: int
    system_efficiency: float
    avg_allocation_time: float
    communication_latency: float
    system_health: float
    convergence_rate: float
    policy_gradient: float

class SystemMonitor:
    """Real-time system monitoring and visualization"""
    
    def __init__(self, config_file: str = "config/system_config.yaml"):
        self.config = ConfigManager(config_file)
        self.logger = setup_logger("system_monitor")
        
        # Monitoring components
        self.ros_interface = ROSInterface("system_monitor")
        self.metrics_collector = MetricsCollector()
        
        # Data storage
        self.status_history = deque(maxlen=1000)
        self.robot_positions = {}
        self.task_locations = {}
        self.performance_metrics = deque(maxlen=500)
        self.alert_queue = queue.Queue()
        
        # Monitoring parameters
        self.update_rate = self.config.get("monitoring.update_rate", 2.0)
        self.enable_visualization = self.config.get("monitoring.enable_visualization", True)
        self.enable_gui = self.config.get("monitoring.enable_gui", True)
        
        # Performance thresholds
        self.thresholds = {
            'efficiency': 0.85,
            'allocation_time': 100.0,  # ms
            'communication_latency': 50.0,  # ms
            'system_health': 0.9,
            'convergence_rate': 0.8
        }
        
        # Visualization setup
        self.fig = None
        self.axes = {}
        self.plots = {}
        self.animation = None
        
        # GUI setup
        self.root = None
        self.gui_widgets = {}
        
        self.logger.info("System monitor initialized")
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        self.logger.info("Starting system monitoring")
        
        # Initialize ROS interface
        await self.ros_interface.initialize()
        
        # Register message handlers
        self.ros_interface.register_message_handler('heartbeat', self.handle_heartbeat)
        self.ros_interface.register_message_handler('task_completion', self.handle_task_completion)
        self.ros_interface.register_message_handler('status_update', self.handle_status_update)
        
        # Start monitoring tasks
        tasks = [
            self.monitor_system_status(),
            self.collect_performance_data(),
            self.analyze_system_health(),
            self.check_alerts()
        ]
        
        # Start visualization if enabled
        if self.enable_visualization:
            visualization_thread = threading.Thread(target=self.start_visualization, daemon=True)
            visualization_thread.start()
        
        # Start GUI if enabled
        if self.enable_gui:
            gui_thread = threading.Thread(target=self.start_gui, daemon=True)
            gui_thread.start()
        
        # Run monitoring tasks
        await asyncio.gather(*tasks)
    
    async def monitor_system_status(self):
        """Monitor overall system status"""
        while True:
            try:
                # Collect current status
                status = await self.collect_system_status()
                
                # Store in history
                self.status_history.append(status)
                
                # Check for alerts
                await self.check_status_alerts(status)
                
                # Log status periodically
                if len(self.status_history) % 30 == 0:  # Every minute at 2Hz
                    self.logger.info(f"System Status: {status.active_robots}/{status.total_robots} robots, "
                                   f"Efficiency: {status.system_efficiency:.3f}, "
                                   f"Health: {status.system_health:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error monitoring system status: {e}")
            
            await asyncio.sleep(1.0 / self.update_rate)
    
    async def collect_system_status(self) -> SystemStatus:
        """Collect current system status"""
        current_time = time.time()
        
        # Get robot status
        active_robots = len([r for r in self.robot_positions.keys() 
                           if self.is_robot_active(r, current_time)])
        total_robots = len(self.robot_positions)
        
        # Calculate performance metrics
        efficiency = self.calculate_system_efficiency()
        avg_allocation_time = self.calculate_avg_allocation_time()
        comm_latency = self.calculate_communication_latency()
        system_health = self.calculate_system_health()
        convergence_rate = self.calculate_convergence_rate()
        policy_gradient = self.calculate_policy_gradient()
        
        # Task statistics (would be retrieved from coordination master)
        pending_tasks = 5  # Placeholder
        completed_tasks = 150  # Placeholder
        failed_tasks = 8  # Placeholder
        
        return SystemStatus(
            timestamp=current_time,
            active_robots=active_robots,
            total_robots=total_robots,
            pending_tasks=pending_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            system_efficiency=efficiency,
            avg_allocation_time=avg_allocation_time,
            communication_latency=comm_latency,
            system_health=system_health,
            convergence_rate=convergence_rate,
            policy_gradient=policy_gradient
        )
    
    def is_robot_active(self, robot_id: str, current_time: float) -> bool:
        """Check if robot is currently active"""
        if robot_id not in self.robot_positions:
            return False
        
        last_update = self.robot_positions[robot_id].get('timestamp', 0)
        return (current_time - last_update) < 10.0  # 10 second timeout
    
    def calculate_system_efficiency(self) -> float:
        """Calculate overall system efficiency"""
        if len(self.performance_metrics) < 2:
            return 0.92  # Default/target efficiency
        
        recent_metrics = list(self.performance_metrics)[-10:]
        efficiencies = [m.get('efficiency', 0.92) for m in recent_metrics]
        return np.mean(efficiencies)
    
    def calculate_avg_allocation_time(self) -> float:
        """Calculate average task allocation time"""
        if len(self.performance_metrics) < 2:
            return 45.0  # Default allocation time in ms
        
        recent_metrics = list(self.performance_metrics)[-10:]
        times = [m.get('allocation_time', 45.0) for m in recent_metrics]
        return np.mean(times)
    
    def calculate_communication_latency(self) -> float:
        """Calculate average communication latency"""
        comm_metrics = self.ros_interface.get_communication_metrics()
        return comm_metrics.get('avg_latency_ms', 20.0)
    
    def calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        if not self.robot_positions:
            return 1.0
        
        # Simple health calculation based on active robots
        active_count = sum(1 for robot_id in self.robot_positions.keys() 
                          if self.is_robot_active(robot_id, time.time()))
        total_count = len(self.robot_positions)
        
        if total_count == 0:
            return 1.0
        
        base_health = active_count / total_count
        
        # Adjust based on communication quality
        comm_metrics = self.ros_interface.get_communication_metrics()
        comm_health = min(1.0, comm_metrics.get('success_rate', 0.95))
        
        return (base_health + comm_health) / 2.0
    
    def calculate_convergence_rate(self) -> float:
        """Calculate Q-learning convergence rate"""
        # Placeholder - would integrate with Q-learning coordinator
        return 0.92
    
    def calculate_policy_gradient(self) -> float:
        """Calculate policy gradient metric"""
        # Placeholder - would integrate with Q-learning coordinator
        return 0.85
    
    async def handle_heartbeat(self, message):
        """Handle robot heartbeat messages"""
        robot_data = message.data
        robot_id = robot_data.get('robot_id')
        
        if robot_id:
            self.robot_positions[robot_id] = {
                'position': robot_data.get('position', (0, 0)),
                'battery_level': robot_data.get('battery_level', 100),
                'status': robot_data.get('status', 'unknown'),
                'timestamp': message.timestamp
            }
    
    async def handle_task_completion(self, message):
        """Handle task completion messages"""
        task_data = message.data
        task_id = task_data.get('task_id')
        success = task_data.get('success', False)
        
        # Record performance metrics
        performance_data = {
            'timestamp': message.timestamp,
            'task_id': task_id,
            'success': success,
            'execution_time': task_data.get('execution_time', 0),
            'energy_used': task_data.get('energy_used', 0)
        }
        
        self.performance_metrics.append(performance_data)
    
    async def handle_status_update(self, message):
        """Handle general status update messages"""
        status_data = message.data
        
        # Store relevant status information
        if 'system_metrics' in status_data:
            metrics = status_data['system_metrics']
            self.performance_metrics.append({
                'timestamp': message.timestamp,
                **metrics
            })
    
    async def collect_performance_data(self):
        """Continuously collect performance data"""
        while True:
            try:
                # Collect metrics from various sources
                comm_metrics = self.ros_interface.get_communication_metrics()
                
                # Store performance data
                perf_data = {
                    'timestamp': time.time(),
                    'communication': comm_metrics,
                    'system_load': self.get_system_load(),
                    'memory_usage': self.get_memory_usage()
                }
                
                await self.metrics_collector.record_metrics(perf_data)
                
            except Exception as e:
                self.logger.error(f"Error collecting performance data: {e}")
            
            await asyncio.sleep(5.0)
    
    def get_system_load(self) -> float:
        """Get system CPU load"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 15.0  # Simulated load
    
    def get_memory_usage(self) -> float:
        """Get system memory usage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 35.0  # Simulated usage
    
    async def analyze_system_health(self):
        """Analyze system health and trends"""
        while True:
            try:
                if len(self.status_history) > 10:
                    # Analyze trends
                    recent_statuses = list(self.status_history)[-10:]
                    
                    # Health trend analysis
                    health_values = [s.system_health for s in recent_statuses]
                    health_trend = np.polyfit(range(len(health_values)), health_values, 1)[0]
                    
                    if health_trend < -0.01:  # Degrading health
                        await self.create_alert("System health degrading", "warning")
                    
                    # Efficiency trend analysis
                    efficiency_values = [s.system_efficiency for s in recent_statuses]
                    efficiency_trend = np.polyfit(range(len(efficiency_values)), efficiency_values, 1)[0]
                    
                    if efficiency_trend < -0.005:  # Degrading efficiency
                        await self.create_alert("System efficiency degrading", "warning")
                
            except Exception as e:
                self.logger.error(f"Error analyzing system health: {e}")
            
            await asyncio.sleep(30.0)  # Analyze every 30 seconds
    
    async def check_status_alerts(self, status: SystemStatus):
        """Check for alert conditions in system status"""
        # Efficiency alert
        if status.system_efficiency < self.thresholds['efficiency']:
            await self.create_alert(
                f"Low system efficiency: {status.system_efficiency:.3f}",
                "warning"
            )
        
        # Allocation time alert
        if status.avg_allocation_time > self.thresholds['allocation_time']:
            await self.create_alert(
                f"High allocation time: {status.avg_allocation_time:.1f}ms",
                "warning"
            )
        
        # Communication latency alert
        if status.communication_latency > self.thresholds['communication_latency']:
            await self.create_alert(
                f"High communication latency: {status.communication_latency:.1f}ms",
                "warning"
            )
        
        # System health alert
        if status.system_health < self.thresholds['system_health']:
            await self.create_alert(
                f"Low system health: {status.system_health:.3f}",
                "critical"
            )
        
        # Robot availability alert
        if status.total_robots > 0:
            availability = status.active_robots / status.total_robots
            if availability < 0.8:
                await self.create_alert(
                    f"Low robot availability: {availability:.1%}",
                    "critical"
                )
    
    async def create_alert(self, message: str, severity: str):
        """Create and queue system alert"""
        alert = {
            'timestamp': time.time(),
            'message': message,
            'severity': severity
        }
        
        self.alert_queue.put(alert)
        
        if severity == "critical":
            self.logger.error(f"CRITICAL ALERT: {message}")
        else:
            self.logger.warning(f"ALERT: {message}")
    
    async def check_alerts(self):
        """Process alert queue"""
        while True:
            try:
                if not self.alert_queue.empty():
                    alert = self.alert_queue.get_nowait()
                    await self.process_alert(alert)
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error processing alerts: {e}")
            
            await asyncio.sleep(1.0)
    
    async def process_alert(self, alert: Dict):
        """Process individual alert"""
        # Log alert
        self.logger.info(f"Processing alert: {alert['message']}")
        
        # Could implement additional alert actions:
        # - Send notifications
        # - Trigger automated responses
        # - Update dashboard
        
        # For now, just store in GUI if available
        if hasattr(self, 'gui_widgets') and 'alerts_text' in self.gui_widgets:
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(alert['timestamp']))
            alert_text = f"[{timestamp_str}] {alert['severity'].upper()}: {alert['message']}\n"
            
            # Update GUI in thread-safe manner
            self.root.after(0, lambda: self.gui_widgets['alerts_text'].insert(tk.END, alert_text))
    
    def start_visualization(self):
        """Start matplotlib visualization"""
        if not self.enable_visualization:
            return
        
        try:
            # Create figure and subplots
            self.fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            self.axes = {
                'robot_map': axes[0, 0],
                'efficiency': axes[0, 1],
                'allocation_time': axes[0, 2],
                'system_health': axes[1, 0],
                'communication': axes[1, 1],
                'convergence': axes[1, 2]
            }
            
            # Setup plots
            self.setup_plots()
            
            # Start animation
            self.animation = animation.FuncAnimation(
                self.fig, self.update_plots, interval=1000, blit=False
            )
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error starting visualization: {e}")
    
    def setup_plots(self):
        """Setup individual plots"""
        # Robot map
        self.axes['robot_map'].set_title('Robot Positions')
        self.axes['robot_map'].set_xlabel('X Position (m)')
        self.axes['robot_map'].set_ylabel('Y Position (m)')
        self.axes['robot_map'].grid(True)
        self.axes['robot_map'].set_xlim(-50, 50)
        self.axes['robot_map'].set_ylim(-50, 50)
        
        # Efficiency plot
        self.axes['efficiency'].set_title('System Efficiency')
        self.axes['efficiency'].set_xlabel('Time')
        self.axes['efficiency'].set_ylabel('Efficiency')
        self.axes['efficiency'].grid(True)
        self.axes['efficiency'].set_ylim(0, 1)
        
        # Allocation time plot
        self.axes['allocation_time'].set_title('Allocation Time')
        self.axes['allocation_time'].set_xlabel('Time')
        self.axes['allocation_time'].set_ylabel('Time (ms)')
        self.axes['allocation_time'].grid(True)
        
        # System health plot
        self.axes['system_health'].set_title('System Health')
        self.axes['system_health'].set_xlabel('Time')
        self.axes['system_health'].set_ylabel('Health Score')
        self.axes['system_health'].grid(True)
        self.axes['system_health'].set_ylim(0, 1)
        
        # Communication plot
        self.axes['communication'].set_title('Communication Latency')
        self.axes['communication'].set_xlabel('Time')
        self.axes['communication'].set_ylabel('Latency (ms)')
        self.axes['communication'].grid(True)
        
        # Convergence plot
        self.axes['convergence'].set_title('Learning Convergence')
        self.axes['convergence'].set_xlabel('Time')
        self.axes['convergence'].set_ylabel('Convergence Rate')
        self.axes['convergence'].grid(True)
        self.axes['convergence'].set_ylim(0, 1)
    
    def update_plots(self, frame):
        """Update all plots with latest data"""
        try:
            # Clear all axes
            for ax in self.axes.values():
                ax.clear()
            
            # Re-setup plots
            self.setup_plots()
            
            # Update robot map
            self.update_robot_map()
            
            # Update time series plots
            self.update_time_series_plots()
            
        except Exception as e:
            self.logger.error(f"Error updating plots: {e}")
    
    def update_robot_map(self):
        """Update robot position map"""
        current_time = time.time()
        
        for robot_id, robot_data in self.robot_positions.items():
            if self.is_robot_active(robot_id, current_time):
                pos = robot_data['position']
                battery = robot_data.get('battery_level', 100)
                status = robot_data.get('status', 'unknown')
                
                # Color based on battery level
                if battery > 50:
                    color = 'green'
                elif battery > 20:
                    color = 'orange'
                else:
                    color = 'red'
                
                # Size based on status
                size = 100 if status == 'active' else 50
                
                self.axes['robot_map'].scatter(pos[0], pos[1], c=color, s=size, alpha=0.7)
                self.axes['robot_map'].annotate(robot_id, (pos[0], pos[1]), 
                                              xytext=(5, 5), textcoords='offset points', 
                                              fontsize=8)
    
    def update_time_series_plots(self):
        """Update time series plots"""
        if len(self.status_history) < 2:
            return
        
        # Get recent data
        recent_data = list(self.status_history)[-50:]  # Last 50 points
        timestamps = [s.timestamp for s in recent_data]
        
        # Convert to relative time (seconds ago)
        current_time = time.time()
        time_ago = [(current_time - t) for t in timestamps]
        
        # Plot efficiency
        efficiency_values = [s.system_efficiency for s in recent_data]
        self.axes['efficiency'].plot(time_ago, efficiency_values, 'b-', linewidth=2)
        self.axes['efficiency'].axhline(y=self.thresholds['efficiency'], 
                                       color='r', linestyle='--', alpha=0.7)
        
        # Plot allocation time
        allocation_times = [s.avg_allocation_time for s in recent_data]
        self.axes['allocation_time'].plot(time_ago, allocation_times, 'g-', linewidth=2)
        self.axes['allocation_time'].axhline(y=self.thresholds['allocation_time'], 
                                           color='r', linestyle='--', alpha=0.7)
        
        # Plot system health
        health_values = [s.system_health for s in recent_data]
        self.axes['system_health'].plot(time_ago, health_values, 'm-', linewidth=2)
        self.axes['system_health'].axhline(y=self.thresholds['system_health'], 
                                         color='r', linestyle='--', alpha=0.7)
        
        # Plot communication latency
        comm_latencies = [s.communication_latency for s in recent_data]
        self.axes['communication'].plot(time_ago, comm_latencies, 'c-', linewidth=2)
        self.axes['communication'].axhline(y=self.thresholds['communication_latency'], 
                                         color='r', linestyle='--', alpha=0.7)
        
        # Plot convergence
        convergence_values = [s.convergence_rate for s in recent_data]
        self.axes['convergence'].plot(time_ago, convergence_values, 'orange', linewidth=2)
        self.axes['convergence'].axhline(y=self.thresholds['convergence_rate'], 
                                       color='r', linestyle='--', alpha=0.7)
    
    def start_gui(self):
        """Start Tkinter GUI"""
        if not self.enable_gui:
            return
        
        try:
            self.root = tk.Tk()
            self.root.title("Multi-Robot Coordination Monitor")
            self.root.geometry("800x600")
            
            # Create GUI elements
            self.create_gui_elements()
            
            # Start GUI update loop
            self.update_gui()
            
            # Start main loop
            self.root.mainloop()
            
        except Exception as e:
            self.logger.error(f"Error starting GUI: {e}")
    
    def create_gui_elements(self):
        """Create GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="10")
        status_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status labels
        self.gui_widgets['active_robots'] = ttk.Label(status_frame, text="Active Robots: -/-")
        self.gui_widgets['active_robots'].grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        self.gui_widgets['efficiency'] = ttk.Label(status_frame, text="Efficiency: -.---")
        self.gui_widgets['efficiency'].grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        self.gui_widgets['health'] = ttk.Label(status_frame, text="Health: -.---")
        self.gui_widgets['health'].grid(row=0, column=2, sticky=tk.W)
        
        self.gui_widgets['allocation_time'] = ttk.Label(status_frame, text="Allocation: --ms")
        self.gui_widgets['allocation_time'].grid(row=1, column=0, sticky=tk.W, padx=(0, 20))
        
        self.gui_widgets['latency'] = ttk.Label(status_frame, text="Latency: --ms")
        self.gui_widgets['latency'].grid(row=1, column=1, sticky=tk.W, padx=(0, 20))
        
        self.gui_widgets['convergence'] = ttk.Label(status_frame, text="Convergence: -.---")
        self.gui_widgets['convergence'].grid(row=1, column=2, sticky=tk.W)
        
        # Alerts frame
        alerts_frame = ttk.LabelFrame(main_frame, text="Alerts", padding="10")
        alerts_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Alerts text area with scrollbar
        self.gui_widgets['alerts_text'] = tk.Text(alerts_frame, height=10, width=80)
        scrollbar = ttk.Scrollbar(alerts_frame, orient=tk.VERTICAL, command=self.gui_widgets['alerts_text'].yview)
        self.gui_widgets['alerts_text'].configure(yscrollcommand=scrollbar.set)
        
        self.gui_widgets['alerts_text'].grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Control buttons
        ttk.Button(control_frame, text="Reset Alerts", command=self.clear_alerts).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(control_frame, text="Export Data", command=self.export_data).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(control_frame, text="Emergency Stop", command=self.emergency_stop).grid(row=0, column=2, padx=(0, 10))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        alerts_frame.columnconfigure(0, weight=1)
        alerts_frame.rowconfigure(0, weight=1)
    
    def update_gui(self):
        """Update GUI with latest data"""
        if not self.root:
            return
        
        try:
            if self.status_history:
                latest_status = self.status_history[-1]
                
                # Update status labels
                self.gui_widgets['active_robots'].config(
                    text=f"Active Robots: {latest_status.active_robots}/{latest_status.total_robots}"
                )
                self.gui_widgets['efficiency'].config(
                    text=f"Efficiency: {latest_status.system_efficiency:.3f}"
                )
                self.gui_widgets['health'].config(
                    text=f"Health: {latest_status.system_health:.3f}"
                )
                self.gui_widgets['allocation_time'].config(
                    text=f"Allocation: {latest_status.avg_allocation_time:.1f}ms"
                )
                self.gui_widgets['latency'].config(
                    text=f"Latency: {latest_status.communication_latency:.1f}ms"
                )
                self.gui_widgets['convergence'].config(
                    text=f"Convergence: {latest_status.convergence_rate:.3f}"
                )
            
            # Schedule next update
            self.root.after(1000, self.update_gui)  # Update every second
            
        except Exception as e:
            self.logger.error(f"Error updating GUI: {e}")
    
    def clear_alerts(self):
        """Clear alerts display"""
        if 'alerts_text' in self.gui_widgets:
            self.gui_widgets['alerts_text'].delete(1.0, tk.END)
    
    def export_data(self):
        """Export monitoring data"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"monitoring_data_{timestamp}.json"
            
            export_data = {
                'status_history': [asdict(s) for s in self.status_history],
                'robot_positions': self.robot_positions,
                'performance_metrics': list(self.performance_metrics),
                'export_timestamp': time.time()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Data exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
    
    def emergency_stop(self):
        """Trigger emergency stop"""
        self.logger.warning("Emergency stop triggered from monitor")
        
        # In a real implementation, this would send emergency stop commands
        # to all robots and the coordination master
        
        # Create critical alert
        asyncio.create_task(self.create_alert("EMERGENCY STOP ACTIVATED", "critical"))
    
    def get_current_status(self) -> Dict:
        """Get current system status for external queries"""
        if self.status_history:
            return asdict(self.status_history[-1])
        return {}


async def main():
    """Main entry point for system monitor"""
    parser = argparse.ArgumentParser(description="Multi-Robot System Monitor")
    parser.add_argument("--config", type=str, default="config/system_config.yaml", help="Config file")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--update-rate", type=float, default=2.0, help="Update rate (Hz)")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = SystemMonitor(args.config)
    
    # Configure options
    if args.no_gui:
        monitor.enable_gui = False
    if args.no_viz:
        monitor.enable_visualization = False
    monitor.update_rate = args.update_rate
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nShutting down system monitor...")
        logging.info("System monitor shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
