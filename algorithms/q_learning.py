#!/usr/bin/env python3
"""
Q-Learning Implementation for Multi-Robot Coordination
Implements Q-learning algorithm for distributed robot coordination
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import time
import logging

class QLearningAgent:
    """Q-Learning agent for individual robot"""
    
    def __init__(self, agent_id: str, learning_rate: float = 0.01, 
                 discount_factor: float = 0.95, exploration_rate: float = 0.15):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.initial_exploration_rate = exploration_rate
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.995
        
        # Q-table: state-action pairs to Q-values
        self.q_table: Dict[Tuple, float] = defaultdict(float)
        
        # State and action spaces
        self.states = set()
        self.actions = ['execute', 'reject', 'defer', 'request_help']
        
        # Performance tracking
        self.update_count = 0
        self.convergence_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=1000)
        
        # Eligibility traces for TD(λ)
        self.eligibility_traces: Dict[Tuple, float] = defaultdict(float)
        self.lambda_trace = 0.9
        
        self.logger = logging.getLogger(f"q_learning_{agent_id}")
        self.logger.info(f"Q-Learning agent initialized for {agent_id}")
    
    def get_state_representation(self, task_type: str, robot_state: Dict = None) -> Tuple:
        """Convert task and robot state to discrete state representation"""
        # Discretize continuous variables for Q-table
        if robot_state:
            battery_level = int(robot_state.get('battery_level', 100) / 10) * 10  # Bins of 10%
            task_load = int(robot_state.get('task_load', 0) * 10)  # Bins of 0.1
            position_x = int(robot_state.get('position', (0, 0))[0] / 5) * 5  # 5m bins
            position_y = int(robot_state.get('position', (0, 0))[1] / 5) * 5  # 5m bins
        else:
            battery_level = 100
            task_load = 0
            position_x = 0
            position_y = 0
        
        state = (task_type, battery_level, task_load, position_x, position_y)
        self.states.add(state)
        return state
    
    def get_q_value(self, state: Tuple, action: str) -> float:
        """Get Q-value for state-action pair"""
        return self.q_table.get((state, action), 0.0)
    
    def select_action(self, state: Tuple, available_actions: List[str] = None) -> str:
        """Select action using epsilon-greedy policy"""
        if available_actions is None:
            available_actions = self.actions
        
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return np.random.choice(available_actions)
        
        # Exploitation: best action
        q_values = [self.get_q_value(state, action) for action in available_actions]
        max_q = max(q_values)
        
        # Handle ties by random selection among best actions
        best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
        return np.random.choice(best_actions)
    
    def update_q_value(self, state: Tuple, action: str, reward: float, 
                      next_state: Optional[Tuple] = None, next_action: Optional[str] = None):
        """Update Q-value using Q-learning or SARSA update rule"""
        current_q = self.get_q_value(state, action)
        
        if next_state is not None:
            if next_action is not None:
                # SARSA update
                next_q = self.get_q_value(next_state, next_action)
            else:
                # Q-learning update (use max Q-value)
                next_q_values = [self.get_q_value(next_state, a) for a in self.actions]
                next_q = max(next_q_values) if next_q_values else 0.0
        else:
            next_q = 0.0  # Terminal state
        
        # TD error
        td_error = reward + self.discount_factor * next_q - current_q
        
        # Update Q-value
        new_q = current_q + self.learning_rate * td_error
        self.q_table[(state, action)] = new_q
        
        # Update eligibility traces
        self.eligibility_traces[(state, action)] += 1.0
        
        # Update all states with eligibility traces
        for (s, a), trace in list(self.eligibility_traces.items()):
            if trace > 0.01:  # Only update significant traces
                self.q_table[(s, a)] += self.learning_rate * td_error * trace
                self.eligibility_traces[(s, a)] *= self.discount_factor * self.lambda_trace
            else:
                del self.eligibility_traces[(s, a)]
        
        # Track metrics
        self.update_count += 1
        self.reward_history.append(reward)
        
        # Calculate convergence metric
        if self.update_count % 10 == 0:
            convergence = self.calculate_convergence()
            self.convergence_history.append(convergence)
        
        self.logger.debug(f"Updated Q({state}, {action}) = {new_q:.3f}, reward = {reward:.3f}")
    
    def decay_exploration_rate(self):
        """Decay exploration rate over time"""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
    
    def calculate_convergence(self) -> float:
        """Calculate convergence metric based on Q-value stability"""
        if len(self.convergence_history) < 2:
            return 0.0
        
        # Calculate variance in recent Q-value updates
        recent_updates = list(self.convergence_history)[-50:]  # Last 50 updates
        if len(recent_updates) < 2:
            return 0.0
        
        variance = np.var(recent_updates)
        convergence = 1.0 / (1.0 + variance)  # Higher variance = lower convergence
        
        return min(1.0, convergence)
    
    def get_policy(self, state: Tuple) -> Dict[str, float]:
        """Get action probabilities for given state"""
        action_values = {action: self.get_q_value(state, action) for action in self.actions}
        
        # Softmax policy
        max_q = max(action_values.values())
        exp_values = {action: np.exp(q - max_q) for action, q in action_values.items()}
        sum_exp = sum(exp_values.values())
        
        if sum_exp == 0:
            # Uniform distribution if all Q-values are the same
            return {action: 1.0 / len(self.actions) for action in self.actions}
        
        return {action: exp_val / sum_exp for action, exp_val in exp_values.items()}
    
    def save_model(self, filepath: str):
        """Save Q-table and agent state to file"""
        model_data = {
            'agent_id': self.agent_id,
            'q_table': dict(self.q_table),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'update_count': self.update_count,
            'states': list(self.states),
            'convergence_history': list(self.convergence_history),
            'reward_history': list(self.reward_history)
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load Q-table and agent state from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = defaultdict(float, model_data['q_table'])
            self.learning_rate = model_data['learning_rate']
            self.discount_factor = model_data['discount_factor']
            self.exploration_rate = model_data['exploration_rate']
            self.update_count = model_data['update_count']
            self.states = set(model_data['states'])
