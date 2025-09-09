#!/usr/bin/env python3
"""
Metrics collection and analysis for Multi-Robot Coordination Framework
"""

import time
import asyncio
import logging
from typing import Dict, List, Any, Optional
from collections import deque, defaultdict
import numpy as np
import json

class MetricsCollector:
    """Collects and analyzes system metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.metric_types = defaultdict(lambda: deque(maxlen=max_history))
        self.logger = logging.getLogger("metrics_collector")
        
        # Performance targets
        self.targets = {
            'efficiency': 0.92,
            'allocation_time': 50.0,  # ms
            'communication_latency': 25.0,  # ms
            'convergence_rate': 0.92,
            'policy_gradient': 0.85,
            'availability': 0.999
        }
    
    async def record_metrics(self, metrics: Dict[str, Any]):
        """Record metrics with timestamp"""
        timestamped_metrics = {
            'timestamp': time.time(),
            **metrics
        }
        
        self.metrics_history.append(timestamped_metrics)
        
        # Store by type for easy analysis
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metric_types[key].append(value)
    
    def get_metric_summary(self, metric_name: str, window_size: Optional[int] = None) -> Dict:
        """Get statistical summary of a metric"""
        if metric_name not in self.metric_types:
            return {}
        
        values = list(self.metric_types[metric_name])
        if window_size:
            values = values[-window_size:]
        
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'count': len(values),
            'latest': values[-1] if values else None,
            'target': self.targets.get(metric_name),
            'meets_target': values[-1] <= self.targets.get(metric_name, float('inf')) if values else False
        }
    
    def get_system_performance_score(self) -> float:
        """Calculate overall system performance score"""
        scores = []
        weights = {
            'efficiency': 0.25,
            'allocation_time': 0.20,
            'communication_latency': 0.15,
            'convergence_rate': 0.20,
            'policy_gradient': 0.10,
            'availability': 0.10
        }
        
        for metric, weight in weights.items():
            summary = self.get_metric_summary(metric, window_size=10)
            if summary and 'latest' in summary and summary['latest'] is not None:
                target = self.targets.get(metric, 1.0)
                
                if metric in ['allocation_time', 'communication_latency']:
                    # Lower is better
                    score = min(1.0, target / max(summary['latest'], 0.001))
                else:
                    # Higher is better
                    score = min(1.0, summary['latest'] / target)
                
                scores.append(score * weight)
        
        return sum(scores) if scores else 0.0


class RobotMetrics:
    """Individual robot metrics"""
    
    def __init__(self, robot_id: str):
        self.robot_id = robot_id
        self.start_time = time.time()
        self.task_metrics = []
        self.performance_history = deque(maxlen=500)
        self.logger = logging.getLogger(f"robot_metrics_{robot_id}")
    
    def record_task_completion(self, task_data: Dict):
        """Record task completion metrics"""
        self.task_metrics.append({
            'timestamp': time.time(),
            'task_id': task_data.get('task_id'),
            'duration': task_data.get('duration', 0),
            'success': task_data.get('success', False),
            'energy_used': task_data.get('energy_used', 0)
        })
    
    def get_robot_performance(self) -> Dict:
        """Get robot performance metrics"""
        if not self.task_metrics:
            return {}
        
        recent_tasks = self.task_metrics[-20:]  # Last 20 tasks
        success_rate = sum(1 for t in recent_tasks if t['success']) / len(recent_tasks)
        avg_duration = np.mean([t['duration'] for t in recent_tasks])
        
        return {
            'robot_id': self.robot_id,
            'total_tasks': len(self.task_metrics),
            'success_rate': success_rate,
            'avg_task_duration': avg_duration,
            'uptime': time.time() - self.start_time
        }
