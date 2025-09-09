#!/usr/bin/env python3
"""
Auction Algorithm for Task Allocation in Multi-Robot Systems
Implements distributed auction-based task allocation with optimization
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import heapq

@dataclass
class Bid:
    """Bid representation for auction"""
    robot_id: str
    task_id: str
    bid_value: float
    capability_match: float
    distance_cost: float
    load_penalty: float
    timestamp: float
    metadata: Dict = None

class AuctionAllocator:
    """Distributed auction algorithm for task allocation"""
    
    def __init__(self, auction_timeout: float = 5.0, min_bid_threshold: float = 0.1):
        self.auction_timeout = auction_timeout
        self.min_bid_threshold = min_bid_threshold
        
        # Auction state
        self.active_auctions: Dict[str, Dict] = {}
        self.bid_history: Dict[str, List[Bid]] = defaultdict(list)
        self.allocation_results: List[Dict] = []
        
        # Performance tracking
        self.successful_allocations = 0
        self.failed_allocations = 0
        self.total_auction_time = 0.0
        self.efficiency_improvements = deque(maxlen=100)
        
        # Optimization parameters
        self.price_increment = 0.05
        self.convergence_threshold = 0.01
        self.max_iterations = 50
        
        self.logger = logging.getLogger("auction_allocator")
        self.logger.info("Auction allocator initialized")
    
    async def run_auction(self, task, available_robots: List[str], 
                         robot_states: Dict) -> Tuple[Optional[str], float]:
        """Run auction for single task allocation"""
        auction_start_time = time.time()
        task_id = task.task_id
        
        self.logger.debug(f"Starting auction for task {task_id}")
        
        # Initialize auction
        auction_data = {
            'task': task,
            'participants': available_robots.copy(),
            'bids': [],
            'current_price': 0.0,
            'iteration': 0,
            'start_time': auction_start_time
        }
        
        self.active_auctions[task_id] = auction_data
        
        try:
            # Collect initial bids from all robots
            initial_bids = await self.collect_bids(task, available_robots, robot_states)
            auction_data['bids'] = initial_bids
            
            if not initial_bids:
                self.logger.warning(f"No bids received for task {task_id}")
                self.failed_allocations += 1
                return None, 0.0
            
            # Run iterative auction process
            winning_robot, winning_bid = await self.iterative_auction(task_id, auction_data)
            
            # Record results
            auction_time = time.time() - auction_start_time
            self.total_auction_time += auction_time
            
            if winning_robot:
                self.successful_allocations += 1
                
                # Calculate efficiency improvement
                efficiency = self.calculate_efficiency_improvement(
                    task, winning_robot, winning_bid, robot_states
                )
                self.efficiency_improvements.append(efficiency)
                
                # Record allocation result
                allocation_result = {
                    'task_id': task_id,
                    'winning_robot': winning_robot,
                    'winning_bid': winning_bid,
                    'auction_time': auction_time,
                    'num_participants': len(available_robots),
                    'num_bids': len(initial_bids),
                    'efficiency_improvement': efficiency
                }
                self.allocation_results.append(allocation_result)
                
                self.logger.info(f"Task {task_id} allocated to {winning_robot} "
                               f"(bid: {winning_bid:.3f}, time: {auction_time*1000:.1f}ms)")
            else:
                self.failed_allocations += 1
                self.logger.warning(f"Auction failed for task {task_id}")
            
            return winning_robot, winning_bid
            
        finally:
            # Clean up auction state
            if task_id in self.active_auctions:
                del self.active_auctions[task_id]
    
    async def collect_bids(self, task, available_robots: List[str], 
                          robot_states: Dict) -> List[Bid]:
        """Collect bids from available robots"""
        bids = []
        
        for robot_id in available_robots:
            if robot_id not in robot_states:
                continue
            
            robot_state = robot_states[robot_id]
            
            # Calculate bid value
            bid_value = self.calculate_bid_value(task, robot_id, robot_state)
            
            if bid_value >= self.min_bid_threshold:
                bid = Bid(
                    robot_id=robot_id,
                    task_id=task.task_id,
                    bid_value=bid_value,
                    capability_match=self.calculate_capability_match(task, robot_state),
                    distance_cost=self.calculate_distance_cost(task, robot_state),
                    load_penalty=self.calculate_load_penalty(robot_state),
                    timestamp=time.time(),
                    metadata={
                        'battery_level': robot_state.battery_level,
                        'position': robot_state.position,
                        'current_load': robot_state.task_load
                    }
                )
                bids.append(bid)
                
                # Store in history
                self.bid_history[task.task_id].append(bid)
        
        # Sort bids by value (highest first)
        bids.sort(key=lambda b: b.bid_value, reverse=True)
        
        self.logger.debug(f"Collected {len(bids)} bids for task {task.task_id}")
        return bids
    
    def calculate_bid_value(self, task, robot_id: str, robot_state) -> float:
        """Calculate bid value for robot-task pair"""
        # Base capability matching score
        capability_score = self.calculate_capability_match(task, robot_state)
        
        # Distance efficiency (closer is better)
        distance_score = 1.0 - self.calculate_distance_cost(task, robot_state)
        
        # Load balancing (less loaded robots bid higher)
        load_score = 1.0 - self.calculate_load_penalty(robot_state)
        
        # Battery level consideration
        battery_score = robot_state.battery_level / 100.0
        
        # Task priority bonus
        priority_score = task.priority if hasattr(task, 'priority') else 1.0
        
        # Combine scores with weights
        weights = {
            'capability': 0.35,
            'distance': 0.25,
            'load': 0.20,
            'battery': 0.15,
            'priority': 0.05
        }
        
        bid_value = (
            weights['capability'] * capability_score +
            weights['distance'] * distance_score +
            weights['load'] * load_score +
            weights['battery'] * battery_score +
            weights['priority'] * priority_score
        )
        
        # Add small random factor to break ties
        bid_value += np.random.uniform(-0.01, 0.01)
        
        return max(0.0, bid_value)
    
    def calculate_capability_match(self, task, robot_state) -> float:
        """Calculate how well robot capabilities match task requirements"""
        if not hasattr(task, 'required_capabilities') or not task.required_capabilities:
            return 1.0  # No specific requirements
        
        robot_capabilities = set(robot_state.capabilities)
        required_capabilities = set(task.required_capabilities)
        
        if not required_capabilities:
            return 1.0
        
        # Calculate intersection over union
        intersection = len(robot_capabilities & required_capabilities)
        union = len(robot_capabilities | required_capabilities)
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / len(required_capabilities)
        
        # Bonus for having extra relevant capabilities
        extra_capabilities = robot_capabilities - required_capabilities
        bonus = min(0.2, len(extra_capabilities) * 0.05)
        
        return min(1.0, jaccard_similarity + bonus)
    
    def calculate_distance_cost(self, task, robot_state) -> float:
        """Calculate normalized distance cost (0-1, lower is better)"""
        if not hasattr(task, 'location'):
            return 0.0  # No location constraint
        
        robot_pos = robot_state.position
        task_pos = task.location
        
        distance = np.sqrt((robot_pos[0] - task_pos[0])**2 + 
                          (robot_pos[1] - task_pos[1])**2)
        
        # Normalize distance (assuming max meaningful distance of 100m)
        max_distance = 100.0
        normalized_distance = min(1.0, distance / max_distance)
        
        return normalized_distance
    
    def calculate_load_penalty(self, robot_state) -> float:
        """Calculate load penalty (0-1, higher load = higher penalty)"""
        return min(1.0, robot_state.task_load)
    
    async def iterative_auction(self, task_id: str, auction_data: Dict) -> Tuple[Optional[str], float]:
        """Run iterative auction process for better allocation"""
        max_iterations = self.max_iterations
        current_iteration = 0
        
        bids = auction_data['bids']
        task = auction_data['task']
        
        if not bids:
            return None, 0.0
        
        # Start with highest bidder
        current_winner = bids[0].robot_id
        current_bid = bids[0].bid_value
        
        # Track price progression
        price_history = [current_bid]
        
        while current_iteration < max_iterations:
            current_iteration += 1
            
            # Check if auction should terminate early
            if current_iteration > 1:
                price_change = abs(price_history[-1] - price_history[-2])
                if price_change < self.convergence_threshold:
                    self.logger.debug(f"Auction converged after {current_iteration} iterations")
                    break
            
            # Find second highest bid for price increment
            if len(bids) > 1:
                second_bid = bids[1].bid_value
                price_increment = max(self.price_increment, 
                                    (current_bid - second_bid) * 0.5)
            else:
                price_increment = self.price_increment
            
            # Update auction price
            new_price = current_bid + price_increment
            auction_data['current_price'] = new_price
            
            # Allow robots to update bids based on new price
            updated_bids = await self.update_bids(task, auction_data, new_price)
            
            if updated_bids:
                updated_bids.sort(key=lambda b: b.bid_value, reverse=True)
                
                new_winner = updated_bids[0].robot_id
                new_bid = updated_bids[0].bid_value
                
                # Check if winner changed or bid improved
                if new_winner != current_winner or new_bid > current_bid:
                    current_winner = new_winner
                    current_bid = new_bid
                    bids = updated_bids
                
                price_history.append(new_bid)
            else:
                # No updated bids, use current winner
                price_history.append(current_bid)
            
            # Add small delay to prevent overwhelming
            await asyncio.sleep(0.001)
        
        auction_data['final_iteration'] = current_iteration
        auction_data['price_history'] = price_history
        
        self.logger.debug(f"Auction for {task_id} completed in {current_iteration} iterations")
        
        return current_winner, current_bid
    
    async def update_bids(self, task, auction_data: Dict, current_price: float) -> List[Bid]:
        """Allow robots to update their bids based on current auction state"""
        updated_bids = []
        
        # In a real implementation, this would send requests to robots
        # For simulation, we'll adjust bids based on competitive pressure
        
        original_bids = auction_data['bids']
        
        for original_bid in original_bids:
            # Calculate updated bid based on competitive pressure
            competitive_factor = min(1.2, current_price / max(original_bid.bid_value, 0.1))
            
            # Add some randomness to simulate different robot strategies
            strategy_factor = np.random.uniform(0.95, 1.05)
            
            updated_bid_value = original_bid.bid_value * competitive_factor * strategy_factor
            
            # Only include if robot can still afford the bid
            if updated_bid_value >= self.min_bid_threshold:
                updated_bid = Bid(
                    robot_id=original_bid.robot_id,
                    task_id=original_bid.task_id,
                    bid_value=updated_bid_value,
                    capability_match=original_bid.capability_match,
                    distance_cost=original_bid.distance_cost,
                    load_penalty=original_bid.load_penalty,
                    timestamp=time.time(),
                    metadata=original_bid.metadata
                )
                updated_bids.append(updated_bid)
        
        return updated_bids
    
    def calculate_efficiency_improvement(self, task, winning_robot: str, 
                                       winning_bid: float, robot_states: Dict) -> float:
        """Calculate efficiency improvement from optimal allocation"""
        if winning_robot not in robot_states:
            return 0.0
        
        robot_state = robot_states[winning_robot]
        
        # Baseline efficiency (random allocation)
        baseline_efficiency = 0.5
        
        # Calculate actual efficiency based on allocation quality
        capability_efficiency = self.calculate_capability_match(task, robot_state)
        distance_efficiency = 1.0 - self.calculate_distance_cost(task, robot_state)
        load_efficiency = 1.0 - self.calculate_load_penalty(robot_state)
        
        actual_efficiency = (capability_efficiency + distance_efficiency + load_efficiency) / 3.0
        
        # Return improvement over baseline
        improvement = (actual_efficiency - baseline_efficiency) / baseline_efficiency
        return max(0.0, improvement)
    
    async def run_multi_task_auction(self, tasks: List, available_robots: List[str], 
                                   robot_states: Dict) -> Dict[str, Tuple[str, float]]:
        """Run auction for multiple tasks simultaneously"""
        self.logger.info(f"Starting multi-task auction for {len(tasks)} tasks")
        
        # Create auction tasks
        auction_tasks = []
        for task in tasks:
            auction_task = asyncio.create_task(
                self.run_auction(task, available_robots.copy(), robot_states)
            )
            auction_tasks.append((task.task_id, auction_task))
        
        # Wait for all auctions to complete
        results = {}
        for task_id, auction_task in auction_tasks:
            try:
                winning_robot, winning_bid = await auction_task
                results[task_id] = (winning_robot, winning_bid)
            except Exception as e:
                self.logger.error(f"Auction failed for task {task_id}: {e}")
                results[task_id] = (None, 0.0)
        
        self.logger.info(f"Multi-task auction completed: {len(results)} results")
        return results
    
    def get_auction_statistics(self) -> Dict:
        """Get comprehensive auction performance statistics"""
        total_auctions = self.successful_allocations + self.failed_allocations
        
        if total_auctions == 0:
            return {
                'total_auctions': 0,
                'success_rate': 0.0,
                'avg_auction_time': 0.0,
                'avg_efficiency_improvement': 0.0
            }
        
        avg_auction_time = self.total_auction_time / total_auctions if total_auctions > 0 else 0.0
        avg_efficiency = np.mean(list(self.efficiency_improvements)) if self.efficiency_improvements else 0.0
        
        # Calculate allocation time statistics
        allocation_times = [result['auction_time'] for result in self.allocation_results]
        
        stats = {
            'total_auctions': total_auctions,
            'successful_allocations': self.successful_allocations,
            'failed_allocations': self.failed_allocations,
            'success_rate': self.successful_allocations / total_auctions,
            'avg_auction_time': avg_auction_time,
            'avg_auction_time_ms': avg_auction_time * 1000,
            'min_auction_time_ms': min(allocation_times) * 1000 if allocation_times else 0.0,
            'max_auction_time_ms': max(allocation_times) * 1000 if allocation_times else 0.0,
            'avg_efficiency_improvement': avg_efficiency,
            'efficiency_improvement_std': np.std(list(self.efficiency_improvements)) if self.efficiency_improvements else 0.0,
            'total_allocation_time': self.total_auction_time,
            'allocation_rate': total_auctions / max(self.total_auction_time, 0.001)  # auctions per second
        }
        
        return stats
    
    def reset_statistics(self):
        """Reset auction statistics"""
        self.successful_allocations = 0
        self.failed_allocations = 0
        self.total_auction_time = 0.0
        self.efficiency_improvements.clear()
        self.allocation_results.clear()
        self.bid_history.clear()
        
        self.logger.info("Auction statistics reset")
    
    def get_bid_analysis(self, task_id: str) -> Dict:
        """Analyze bidding patterns for a specific task"""
        if task_id not in self.bid_history:
            return {}
        
        bids = self.bid_history[task_id]
        
        if not bids:
            return {}
        
        bid_values = [bid.bid_value for bid in bids]
        capability_matches = [bid.capability_match for bid in bids]
        distance_costs = [bid.distance_cost for bid in bids]
        
        analysis = {
            'num_bids': len(bids),
            'winning_bid': max(bid_values),
            'avg_bid': np.mean(bid_values),
            'bid_std': np.std(bid_values),
            'min_bid': min(bid_values),
            'max_bid': max(bid_values),
            'avg_capability_match': np.mean(capability_matches),
            'avg_distance_cost': np.mean(distance_costs),
            'bid_range': max(bid_values) - min(bid_values),
            'competition_level': np.std(bid_values) / max(np.mean(bid_values), 0.001)
        }
        
        return analysis
    
    async def optimize_auction_parameters(self, performance_history: List[Dict]):
        """Optimize auction parameters based on performance history"""
        if len(performance_history) < 10:
            return
        
        recent_performance = performance_history[-10:]
        
        # Analyze auction time trends
        avg_time = np.mean([p['avg_auction_time_ms'] for p in recent_performance])
        time_variance = np.var([p['avg_auction_time_ms'] for p in recent_performance])
        
        # Analyze efficiency trends
        avg_efficiency = np.mean([p['avg_efficiency_improvement'] for p in recent_performance])
        
        # Adjust parameters based on performance
        if avg_time > 100.0:  # If auctions are taking too long
            self.auction_timeout = max(1.0, self.auction_timeout * 0.9)
            self.max_iterations = max(10, int(self.max_iterations * 0.9))
            self.logger.info("Reduced auction timeout and iterations to improve speed")
        
        elif avg_time < 20.0 and avg_efficiency < 0.2:  # If too fast but inefficient
            self.auction_timeout = min(10.0, self.auction_timeout * 1.1)
            self.max_iterations = min(100, int(self.max_iterations * 1.1))
            self.logger.info("Increased auction timeout and iterations to improve efficiency")
        
        # Adjust bid threshold based on competition level
        competition_levels = []
        for result in self.allocation_results[-20:]:  # Last 20 allocations
            task_id = result['task_id']
            analysis = self.get_bid_analysis(task_id)
            if 'competition_level' in analysis:
                competition_levels.append(analysis['competition_level'])
        
        if competition_levels:
            avg_competition = np.mean(competition_levels)
            
            if avg_competition < 0.1:  # Low competition
                self.min_bid_threshold = max(0.05, self.min_bid_threshold * 0.95)
            elif avg_competition > 0.5:  # High competition
                self.min_bid_threshold = min(0.3, self.min_bid_threshold * 1.05)


class DistributedAuctioneer:
    """Distributed auction coordinator for scalable task allocation"""
    
    def __init__(self, region_id: str):
        self.region_id = region_id
        self.local_allocator = AuctionAllocator()
        self.neighboring_regions: Set[str] = set()
        self.cross_region_tasks: Dict[str, Dict] = {}
        
        self.logger = logging.getLogger(f"distributed_auctioneer_{region_id}")
    
    def add_neighboring_region(self, region_id: str):
        """Add neighboring region for cross-region coordination"""
        self.neighboring_regions.add(region_id)
        self.logger.info(f"Added neighboring region: {region_id}")
    
    async def coordinate_cross_region_allocation(self, task, local_robots: List[str], 
                                               robot_states: Dict) -> Tuple[Optional[str], float]:
        """Coordinate task allocation across regions"""
        # First try local allocation
        local_winner, local_bid = await self.local_allocator.run_auction(
            task, local_robots, robot_states
        )
        
        # If local allocation is not satisfactory, try cross-region
        if local_bid < 0.7:  # Threshold for cross-region consideration
            cross_region_winner, cross_region_bid = await self.request_cross_region_bids(task)
            
            if cross_region_bid > local_bid:
                return cross_region_winner, cross_region_bid
        
        return local_winner, local_bid
    
    async def request_cross_region_bids(self, task) -> Tuple[Optional[str], float]:
        """Request bids from neighboring regions"""
        # This would implement inter-region communication protocol
        # For simulation, we'll return a random bid
        
        if self.neighboring_regions:
            # Simulate cross-region bidding
            cross_region_bid = np.random.uniform(0.3, 0.8)
            cross_region_robot = f"cross_region_robot_{np.random.randint(1, 100)}"
            
            self.logger.debug(f"Cross-region bid received: {cross_region_bid:.3f}")
            return cross_region_robot, cross_region_bid
        
        return None, 0.0


if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    from dataclasses import dataclass
    
    @dataclass
    class MockTask:
        task_id: str
        task_type: str
        location: Tuple[float, float]
        required_capabilities: List[str]
        priority: float = 1.0
    
    @dataclass
    class MockRobotState:
        robot_id: str
        position: Tuple[float, float]
        capabilities: List[str]
        battery_level: float
        task_load: float
        status: str = "active"
    
    async def test_auction_system():
        """Test the auction allocation system"""
        print("Testing Multi-Robot Auction Allocation System")
        
        # Create auction allocator
        allocator = AuctionAllocator(auction_timeout=2.0)
        
        # Create mock tasks
        tasks = [
            MockTask("task_1", "navigation", (10.0, 20.0), ["navigation"], 1.5),
            MockTask("task_2", "manipulation", (5.0, 15.0), ["manipulation"], 1.0),
            MockTask("task_3", "sensing", (15.0, 25.0), ["sensing"], 2.0),
        ]
        
        # Create mock robot states
        robot_states = {
            "robot_1": MockRobotState("robot_1", (0.0, 0.0), ["navigation", "sensing"], 85.0, 0.2),
            "robot_2": MockRobotState("robot_2", (8.0, 12.0), ["manipulation"], 90.0, 0.1),
            "robot_3": MockRobotState("robot_3", (12.0, 18.0), ["navigation", "manipulation"], 75.0, 0.4),
            "robot_4": MockRobotState("robot_4", (20.0, 30.0), ["sensing", "analysis"], 95.0, 0.0),
        }
        
        available_robots = list(robot_states.keys())
        
        print(f"\nRunning auctions for {len(tasks)} tasks with {len(available_robots)} robots")
        
        # Run individual auctions
        for task in tasks:
            print(f"\n--- Auction for {task.task_id} ---")
            winner, bid = await allocator.run_auction(task, available_robots.copy(), robot_states)
            
            if winner:
                print(f"Winner: {winner} with bid {bid:.3f}")
                
                # Analyze bidding
                analysis = allocator.get_bid_analysis(task.task_id)
                print(f"Bid analysis: {analysis}")
            else:
                print("No winner found")
        
        # Get overall statistics
        stats = allocator.get_auction_statistics()
        print(f"\n--- Overall Auction Statistics ---")
        for key, value in stats.items():
            print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
        
        # Test multi-task auction
        print(f"\n--- Multi-Task Auction ---")
        results = await allocator.run_multi_task_auction(tasks, available_robots, robot_states)
        
        for task_id, (winner, bid) in results.items():
            print(f"{task_id}: {winner} (bid: {bid:.3f})")
    
    # Run the test
    asyncio.run(test_auction_system())
