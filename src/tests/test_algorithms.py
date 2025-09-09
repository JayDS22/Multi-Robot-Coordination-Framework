
#!/usr/bin/env python3
"""
Unit tests for algorithm implementations
"""

import pytest
import numpy as np
import asyncio
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.q_learning import QLearningAgent, QLearningCoordinator
from algorithms.auction_algorithm import AuctionAllocator, Bid
from algorithms.policy_gradient import REINFORCEAgent, ActorCriticAgent

class TestQLearningAgent:
    """Test cases for Q-Learning Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create a test Q-Learning agent"""
        return QLearningAgent("test_agent", learning_rate=0.1, exploration_rate=0.2)
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.agent_id == "test_agent"
        assert agent.learning_rate == 0.1
        assert agent.exploration_rate == 0.2
        assert len(agent.q_table) == 0
        assert len(agent.states) == 0
    
    def test_state_representation(self, agent):
        """Test state representation"""
        robot_state = {
            'battery_level': 75.0,
            'task_load': 0.3,
            'position': (10, 15)
        }
        
        state = agent.get_state_representation("navigation", robot_state)
        
        assert isinstance(state, tuple)
        assert state[0] == "navigation"  # task_type
        assert state[1] == 70  # battery level (binned)
        assert len(state) == 5  # All state components
    
    def test_q_value_operations(self, agent):
        """Test Q-value get/set operations"""
        state = ("navigation", 100, 0, 0, 0)
        action = "execute"
        
        # Initial Q-value should be 0
        initial_q = agent.get_q_value(state, action)
        assert initial_q == 0.0
        
        # Update Q-value
        reward = 1.0
        agent.update_q_value(state, action, reward)
        
        # Q-value should have changed
        updated_q = agent.get_q_value(state, action)
        assert updated_q != initial_q
        assert updated_q > 0  # Should be positive for positive reward
    
    def test_action_selection(self, agent):
        """Test action selection with epsilon-greedy"""
        state = ("pickup", 80, 2, 5, 10)
        
        # Test with available actions
        available_actions = ["execute", "reject", "defer"]
        action = agent.select_action(state, available_actions)
        
        assert action in available_actions
        assert isinstance(action, str)
    
    def test_exploration_decay(self, agent):
        """Test exploration rate decay"""
        initial_exploration = agent.exploration_rate
        
        # Decay multiple times
        for _ in range(10):
            agent.decay_exploration_rate()
        
        assert agent.exploration_rate < initial_exploration
        assert agent.exploration_rate >= agent.min_exploration_rate
    
    def test_convergence_calculation(self, agent):
        """Test convergence calculation"""
        # Initially should return 0 (no history)
        convergence = agent.calculate_convergence()
        assert convergence == 0.0
        
        # Add some updates to build history
        state = ("test", 100, 0, 0, 0)
        for i in range(20):
            agent.update_q_value(state, "execute", 0.5 + np.random.normal(0, 0.1))
        
        convergence = agent.calculate_convergence()
        assert isinstance(convergence, float)
        assert 0.0 <= convergence <= 1.0
    
    def test_policy_calculation(self, agent):
        """Test policy calculation"""
        state = ("navigation", 90, 1, 0, 5)
        
        # Train agent a bit
        for action in agent.actions:
            agent.update_q_value(state, action, np.random.uniform(-0.5, 1.0))
        
        policy = agent.get_policy(state)
        
        assert isinstance(policy, dict)
        assert len(policy) == len(agent.actions)
        assert all(0 <= prob <= 1 for prob in policy.values())
        assert abs(sum(policy.values()) - 1.0) < 0.001  # Should sum to 1

class TestQLearningCoordinator:
    """Test cases for Q-Learning Coordinator"""
    
    @pytest.fixture
    def coordinator(self):
        """Create a test Q-Learning coordinator"""
        return QLearningCoordinator()
    
    def test_coordinator_initialization(self, coordinator):
        """Test coordinator initialization"""
        assert len(coordinator.agents) == 0
        assert len(coordinator.global_q_table) == 0
        assert coordinator.learning_rate > 0
        assert coordinator.exploration_rate > 0
    
    def test_agent_management(self, coordinator):
        """Test adding and removing agents"""
        # Add agent
        agent = coordinator.add_agent("test_agent_1")
        assert "test_agent_1" in coordinator.agents
        assert isinstance(agent, QLearningAgent)
        
        # Add another agent
        coordinator.add_agent("test_agent_2")
        assert len(coordinator.agents) == 2
        
        # Remove agent
        coordinator.remove_agent("test_agent_1")
        assert "test_agent_1" not in coordinator.agents
        assert len(coordinator.agents) == 1
    
    def test_coordination_state_generation(self, coordinator):
        """Test coordination state generation"""
        # Mock robots and tasks data
        robots = {
            "robot_1": type('Robot', (), {'status': 'active', 'battery_level': 80, 'task_load': 0.2}),
            "robot_2": type('Robot', (), {'status': 'active', 'battery_level': 60, 'task_load': 0.5})
        }
        
        tasks = {
            "task_1": type('Task', (), {'status': 'pending'}),
            "task_2": type('Task', (), {'status': 'assigned'})
        }
        
        state = coordinator.get_coordination_state(robots, tasks)
        
        assert isinstance(state, tuple)
        assert len(state) == 4  # Expected state components
        assert all(isinstance(x, (int, float)) for x in state)
    
    def test_coordination_action_selection(self, coordinator):
        """Test coordination action selection"""
        state = (5, 3, 80, 2)  # Mock coordination state
        
        action = coordinator.select_coordination_action(state)
        assert action in coordinator.coordination_actions
    
    def test_cooperation_reward_calculation(self, coordinator):
        """Test cooperation reward calculation"""
        # Add some agents
        coordinator.add_agent("agent_1")
        coordinator.add_agent("agent_2")
        
        nearby_agents = ["agent_2"]
        reward = coordinator.calculate_cooperation_reward("agent_1", "navigation", nearby_agents)
        
        assert isinstance(reward, float)
        assert reward >= 0  # Cooperation should give positive reward

class TestAuctionAllocator:
    """Test cases for Auction Algorithm"""
    
    @pytest.fixture
    def allocator(self):
        """Create a test auction allocator"""
        return AuctionAllocator(auction_timeout=2.0, min_bid_threshold=0.1)
    
    @pytest.fixture
    def mock_task(self):
        """Create a mock task for testing"""
        return type('Task', (), {
            'task_id': 'test_task_001',
            'task_type': 'pickup',
            'priority': 1.5,
            'location': (10, 20),
            'required_capabilities': ['navigation', 'manipulation'],
            'estimated_duration': 30.0
        })
    
    @pytest.fixture
    def mock_robot_states(self):
        """Create mock robot states"""
        return {
            'robot_1': type('Robot', (), {
                'position': (5, 15),
                'capabilities': ['navigation', 'manipulation'],
                'battery_level': 80.0,
                'task_load': 0.2
            }),
            'robot_2': type('Robot', (), {
                'position': (15, 25),
                'capabilities': ['navigation', 'sensing'],
                'battery_level': 60.0,
                'task_load': 0.5
            }),
            'robot_3': type('Robot', (), {
                'position': (8, 18),
                'capabilities': ['navigation', 'manipulation', 'sensing'],
                'battery_level': 90.0,
                'task_load': 0.1
            })
        }
    
    def test_allocator_initialization(self, allocator):
        """Test auction allocator initialization"""
        assert allocator.auction_timeout == 2.0
        assert allocator.min_bid_threshold == 0.1
        assert len(allocator.active_auctions) == 0
        assert allocator.successful_allocations == 0
    
    def test_capability_matching(self, allocator, mock_task, mock_robot_states):
        """Test capability matching calculation"""
        robot_state = mock_robot_states['robot_1']
        match_score = allocator.calculate_capability_match(mock_task, robot_state)
        
        assert isinstance(match_score, float)
        assert 0.0 <= match_score <= 1.0
        
        # Robot with exact capabilities should score well
        assert match_score > 0.8
    
    def test_distance_cost_calculation(self, allocator, mock_task, mock_robot_states):
        """Test distance cost calculation"""
        robot_state = mock_robot_states['robot_1']
        distance_cost = allocator.calculate_distance_cost(mock_task, robot_state)
        
        assert isinstance(distance_cost, float)
        assert 0.0 <= distance_cost <= 1.0
    
    def test_bid_value_calculation(self, allocator, mock_task, mock_robot_states):
        """Test bid value calculation"""
        robot_state = mock_robot_states['robot_1']
        bid_value = allocator.calculate_bid_value(mock_task, 'robot_1', robot_state)
        
        assert isinstance(bid_value, float)
        assert bid_value >= 0.0
    
    @pytest.mark.asyncio
    async def test_bid_collection(self, allocator, mock_task, mock_robot_states):
        """Test bid collection from robots"""
        available_robots = list(mock_robot_states.keys())
        bids = await allocator.collect_bids(mock_t
