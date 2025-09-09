#!/usr/bin/env python3
"""
Policy Gradient Methods for Multi-Robot Coordination
Implements REINFORCE and Actor-Critic algorithms
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

class PolicyNetwork(nn.Module):
    """Neural network for policy approximation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class ValueNetwork(nn.Module):
    """Value function approximation for Actor-Critic"""
    
    def __init__(self, state_dim: int, hidden_size: int = 128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        value = self.fc3(x)
        return value

class REINFORCEAgent:
    """REINFORCE policy gradient agent"""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Policy network
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Performance tracking
        self.episode_returns = deque(maxlen=100)
        self.policy_losses = deque(maxlen=100)
        
        self.logger = logging.getLogger("reinforce_agent")
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
        
        # Sample action from probability distribution
        action = torch.multinomial(action_probs, 1).item()
        
        # Store for training
        self.episode_states.append(state)
        self.episode_actions.append(action)
        
        return action
    
    def store_reward(self, reward: float):
        """Store reward for current step"""
        self.episode_rewards.append(reward)
    
    def update_policy(self, gamma: float = 0.95):
        """Update policy using REINFORCE algorithm"""
        if len(self.episode_rewards) == 0:
            return
        
        # Calculate returns
        returns = self.calculate_returns(gamma)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.episode_states))
        actions = torch.LongTensor(self.episode_actions)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        action_probs = self.policy_net(states)
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # REINFORCE loss: -log(π(a|s)) * G_t
        policy_loss = -torch.log(selected_action_probs + 1e-8) * returns
        policy_loss = policy_loss.mean()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Store metrics
        episode_return = sum(self.episode_rewards)
        self.episode_returns.append(episode_return)
        self.policy_losses.append(policy_loss.item())
        
        # Clear episode memory
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        
        self.logger.debug(f"Policy updated - Loss: {policy_loss.item():.4f}, Return: {episode_return:.2f}")
    
    def calculate_returns(self, gamma: float) -> List[float]:
        """Calculate discounted returns for episode"""
        returns = []
        G = 0
        
        # Calculate returns backwards
        for reward in reversed(self.episode_rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        return returns
    
    def get_average_return(self) -> float:
        """Get average return over recent episodes"""
        if len(self.episode_returns) == 0:
            return 0.0
        return np.mean(list(self.episode_returns))

class ActorCriticAgent:
    """Actor-Critic policy gradient agent"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 actor_lr: float = 0.001, critic_lr: float = 0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor network (policy)
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic network (value function)
        self.critic = ValueNetwork(state_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        
        # Performance tracking
        self.episode_returns = deque(maxlen=100)
        self.actor_losses = deque(maxlen=100)
        self.critic_losses = deque(maxlen=100)
        
        self.logger = logging.getLogger("actor_critic_agent")
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using actor network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action probabilities and value estimate
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
        
        # Sample action
        action = torch.multinomial(action_probs, 1).item()
        
        # Store for training
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_values.append(value.item())
        
        return action
    
    def store_reward(self, reward: float):
        """Store reward for current step"""
        self.episode_rewards.append(reward)
    
    def update_networks(self, gamma: float = 0.95):
        """Update both actor and critic networks"""
        if len(self.episode_rewards) == 0:
            return
        
        # Calculate returns and advantages
        returns = self.calculate_returns(gamma)
        advantages = self.calculate_advantages(returns)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.episode_states))
        actions = torch.LongTensor(self.episode_actions)
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Update critic (value function)
        values = self.critic(states).squeeze()
        critic_loss = F.mse_loss(values, returns_tensor)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Update actor (policy)
        action_probs = self.actor(states)
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Actor loss: -log(π(a|s)) * A(s,a)
        actor_loss = -torch.log(selected_action_probs + 1e-8) * advantages_tensor
        actor_loss = actor_loss.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Store metrics
        episode_return = sum(self.episode_rewards)
        self.episode_returns.append(episode_return)
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        # Clear episode memory
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_values.clear()
        
        self.logger.debug(f"Networks updated - Actor Loss: {actor_loss.item():.4f}, "
                         f"Critic Loss: {critic_loss.item():.4f}, Return: {episode_return:.2f}")
    
    def calculate_returns(self, gamma: float) -> List[float]:
        """Calculate discounted returns"""
        returns = []
        G = 0
        
        for reward in reversed(self.episode_rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        return returns
    
    def calculate_advantages(self, returns: List[float]) -> List[float]:
        """Calculate advantages using returns and value estimates"""
        advantages = []
        
        for i, (G, V) in enumerate(zip(returns, self.episode_values)):
            advantage = G - V
            advantages.append(advantage)
        
        return advantages
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        metrics = {
            'avg_return': np.mean(list(self.episode_returns)) if self.episode_returns else 0.0,
            'avg_actor_loss': np.mean(list(self.actor_losses)) if self.actor_losses else 0.0,
            'avg_critic_loss': np.mean(list(self.critic_losses)) if self.critic_losses else 0.0,
            'return_std': np.std(list(self.episode_returns)) if self.episode_returns else 0.0
        }
        
        return metrics

class MultiAgentPolicyGradient:
    """Multi-agent policy gradient coordinator"""
    
    def __init__(self, num_agents: int, state_dim: int, action_dim: int, 
                 algorithm: str = "actor_critic"):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm
        
        # Create agents
        self.agents = {}
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            if algorithm == "reinforce":
                self.agents[agent_id] = REINFORCEAgent(state_dim, action_dim)
            else:  # actor_critic
                self.agents[agent_id] = ActorCriticAgent(state_dim, action_dim)
        
        # Global metrics
        self.global_returns = deque(maxlen=100)
        self.policy_gradients = deque(maxlen=100)
        
        self.logger = logging.getLogger("multi_agent_pg")
        self.logger.info(f"Multi-agent policy gradient initialized with {num_agents} agents")
    
    def select_actions(self, states: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Select actions for all agents"""
        actions = {}
        
        for agent_id, state in states.items():
            if agent_id in self.agents:
                action = self.agents[agent_id].select_action(state)
                actions[agent_id] = action
        
        return actions
    
    def store_rewards(self, rewards: Dict[str, float]):
        """Store rewards for all agents"""
        for agent_id, reward in rewards.items():
            if agent_id in self.agents:
                self.agents[agent_id].store_reward(reward)
    
    def update_all_agents(self, gamma: float = 0.95):
        """Update all agent policies"""
        total_return = 0
        
        for agent_id, agent in self.agents.items():
            if self.algorithm == "reinforce":
                agent.update_policy(gamma)
                total_return += agent.get_average_return()
            else:  # actor_critic
                agent.update_networks(gamma)
                metrics = agent.get_performance_metrics()
                total_return += metrics['avg_return']
        
        # Store global metrics
        avg_return = total_return / len(self.agents)
        self.global_returns.append(avg_return)
        
        # Calculate policy gradient metric
        policy_gradient = self.calculate_policy_gradient()
        self.policy_gradients.append(policy_gradient)
        
        self.logger.debug(f"Updated all agents - Avg Return: {avg_return:.2f}, "
                         f"Policy Gradient: {policy_gradient:.4f}")
    
    def calculate_policy_gradient(self) -> float:
        """Calculate policy gradient convergence metric"""
        if len(self.global_returns) < 10:
            return 0.0
        
        # Calculate gradient of recent returns
        recent_returns = list(self.global_returns)[-10:]
        x = np.arange(len(recent_returns))
        
        # Linear regression to get slope (gradient)
        if len(recent_returns) > 1:
            gradient = np.polyfit(x, recent_returns, 1)[0]
            # Normalize to 0-1 range (0.85 target)
            normalized_gradient = min(1.0, max(0.0, (gradient + 1.0) / 2.0))
            return normalized_gradient
        
        return 0.0
    
    def get_system_performance(self) -> Dict[str, float]:
        """Get overall system performance"""
        if not self.agents:
            return {}
        
        # Aggregate agent performance
        total_return = 0
        total_loss = 0
        
        for agent in self.agents.values():
            if self.algorithm == "reinforce":
                total_return += agent.get_average_return()
                if agent.policy_losses:
                    total_loss += np.mean(list(agent.policy_losses))
            else:  # actor_critic
                metrics = agent.get_performance_metrics()
                total_return += metrics['avg_return']
                total_loss += metrics['avg_actor_loss']
        
        avg_return = total_return / len(self.agents)
        avg_loss = total_loss / len(self.agents)
        
        return {
            'avg_return': avg_return,
            'avg_loss': avg_loss,
            'policy_gradient': self.calculate_policy_gradient(),
            'num_agents': len(self.agents),
            'episodes_trained': len(self.global_returns)
        }
    
    def save_models(self, filepath_prefix: str):
        """Save all agent models"""
        for agent_id, agent in self.agents.items():
            if self.algorithm == "reinforce":
                torch.save(agent.policy_net.state_dict(), f"{filepath_prefix}_{agent_id}_policy.pth")
            else:  # actor_critic
                torch.save(agent.actor.state_dict(), f"{filepath_prefix}_{agent_id}_actor.pth")
                torch.save(agent.critic.state_dict(), f"{filepath_prefix}_{agent_id}_critic.pth")
        
        self.logger.info(f"Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str):
        """Load all agent models"""
        for agent_id, agent in self.agents.items():
            try:
                if self.algorithm == "reinforce":
                    agent.policy_net.load_state_dict(
                        torch.load(f"{filepath_prefix}_{agent_id}_policy.pth")
                    )
                else:  # actor_critic
                    agent.actor.load_state_dict(
                        torch.load(f"{filepath_prefix}_{agent_id}_actor.pth")
                    )
                    agent.critic.load_state_dict(
                        torch.load(f"{filepath_prefix}_{agent_id}_critic.pth")
                    )
            except FileNotFoundError:
                self.logger.warning(f"Model file not found for {agent_id}")
        
        self.logger.info(f"Models loaded with prefix: {filepath_prefix}")


if __name__ == "__main__":
    # Test policy gradient implementation
    import matplotlib.pyplot as plt
    
    def test_policy_gradient():
        # Create multi-agent system
        num_agents = 3
        state_dim = 10
        action_dim = 4
        
        pg_system = MultiAgentPolicyGradient(num_agents, state_dim, action_dim, "actor_critic")
        
        # Simulate training episodes
        num_episodes = 100
        returns_history = []
        
        for episode in range(num_episodes):
            # Generate random states
            states = {
                f"agent_{i}": np.random.randn(state_dim)
                for i in range(num_agents)
            }
            
            # Select actions
            actions = pg_system.select_actions(states)
            
            # Simulate rewards (higher for cooperative behavior)
            rewards = {
                agent_id: np.random.randn() + 0.1 * episode / num_episodes
                for agent_id in actions.keys()
            }
            
            # Store rewards
            pg_system.store_rewards(rewards)
            
            # Update policies at end of episode
            pg_system.update_all_agents()
            
            # Track performance
            performance = pg_system.get_system_performance()
            returns_history.append(performance.get('avg_return', 0))
            
            if episode % 10 == 0:
                print(f"Episode {episode}: Avg Return = {performance.get('avg_return', 0):.3f}, "
                      f"Policy Gradient = {performance.get('policy_gradient', 0):.3f}")
        
        # Plot results
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(returns_history)
        plt.title('Average Return Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Average Return')
        
        plt.subplot(1, 2, 2)
        policy_gradients = list(pg_system.policy_gradients)
        plt.plot(policy_gradients)
        plt.title('Policy Gradient Convergence')
        plt.xlabel('Episode')
        plt.ylabel('Policy Gradient')
        plt.axhline(y=0.85, color='r', linestyle='--', label='Target (0.85)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Final performance
        final_performance = pg_system.get_system_performance()
        print(f"\nFinal Performance:")
        for key, value in final_performance.items():
            print(f"{key}: {value:.4f}")
    
    test_policy_gradient()
