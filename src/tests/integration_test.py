#!/usr/bin/env python3
"""
Integration tests for Multi-Robot Coordination Framework
Tests the complete system with multiple components
"""

import asyncio
import pytest
import logging
import time
import tempfile
import os
from typing import Dict, List
import numpy as np

# Import framework components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from coordination_master import CoordinationMaster
from robot_agent import RobotAgent
from task_generator import TaskGenerator
from algorithms.q_learning import QLearningCoordinator
from algorithms.auction_algorithm import AuctionAllocator
from communication.fault_tolerance import FaultToleranceManager
from utils.config import ConfigManager

class IntegrationTestSuite:
    """Complete integration test suite"""
    
    def __init__(self):
        self.logger = logging.getLogger("integration_test")
        self.test_results = {}
        
        # Test configuration
        self.num_robots = 3
        self.test_duration = 30.0  # seconds
        self.task_generation_rate = 1.0  # tasks per second
        
        # Components
        self.master = None
        self.agents = []
        self.task_generator = None
        
        # Performance targets
        self.performance_targets = {
            'efficiency': 0.85,
            'allocation_time': 100.0,  # ms
            'convergence_rate': 0.80,
            'availability': 0.95,
            'communication_success': 0.95
        }
    
    async def setup_test_environment(self):
        """Setup test environment with all components"""
        self.logger.info("Setting up integration test environment")
        
        # Create temporary config
        config_data = {
            'coordination': {
                'max_robots': 10,
                'heartbeat_interval': 0.5,
                'task_timeout': 20.0
            },
            'learning': {
                'learning_rate': 0.02,
                'exploration_rate': 0.2
            },
            'fault_tolerance': {
                'max_retries': 2,
                'failover_threshold': 1.0
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            # Initialize coordination master
            self.master = CoordinationMaster(config_file)
            
            # Initialize robot agents
            for i in range(self.num_robots):
                agent = RobotAgent(f"test_robot_{i}", config_file)
                self.agents.append(agent)
            
            # Initialize task generator
            self.task_generator = TaskGenerator(config_file)
            self.task_generator.set_generation_rate(self.task_generation_rate)
            self.task_generator.set_environment("warehouse")
            
            self.logger.info(f"Test environment setup complete with {self.num_robots} robots")
            
        finally:
            # Clean up config file
            os.unlink(config_file)
    
    async def test_basic_coordination(self) -> Dict:
        """Test basic coordination functionality"""
        self.logger.info("Testing basic coordination")
        
        results = {
            'test_name': 'basic_coordination',
            'success': False,
            'metrics': {},
            'errors': []
        }
        
        try:
            # Register robots with master
            for agent in self.agents:
                success = await self.master.register_robot(
                    agent.robot_id,
                    agent.state.capabilities,
                    agent.state.position
                )
                if not success:
                    results['errors'].append(f"Failed to register {agent.robot_id}")
            
            # Generate and submit test tasks
            tasks = []
            for i in range(5):
                task = self.task_generator.generate_task()
                await self.master.submit_task(task)
                tasks.append(task)
            
            # Wait for task allocation
            await asyncio.sleep(2.0)
            
            # Check allocation results
            allocated_tasks = 0
            for task in tasks:
                if task.task_id in self.master.tasks:
                    master_task = self.master.tasks[task.task_id]
                    if master_task.status == "assigned":
                        allocated_tasks += 1
            
            allocation_rate = allocated_tasks / len(tasks)
            
            results['metrics'] = {
                'tasks_generated': len(tasks),
                'tasks_allocated': allocated_tasks,
                'allocation_rate': allocation_rate,
                'active_robots': len([r for r in self.master.robots.values() if r.status == "active"])
            }
            
            # Success criteria
            results['success'] = allocation_rate >= 0.8
            
        except Exception as e:
            results['errors'].append(str(e))
            self.logger.error(f"Basic coordination test failed: {e}")
        
        return results
    
    async def test_q_learning_convergence(self) -> Dict:
        """Test Q-learning algorithm convergence"""
        self.logger.info("Testing Q-learning convergence")
        
        results = {
            'test_name': 'q_learning_convergence',
            'success': False,
            'metrics': {},
            'errors': []
        }
        
        try:
            # Create Q-learning coordinator
            coordinator = QLearningCoordinator()
            
            # Add agents
            for agent in self.agents:
                coordinator.add_agent(agent.robot_id)
            
            # Simulate learning episodes
            num_episodes = 50
            convergence_history = []
            
            for episode in range(num_episodes):
                # Simulate allocation results
                allocation_results = []
                for agent in self.agents:
                    result = {
                        'agent_id': agent.robot_id,
                        'task_type': np.random.choice(['pickup', 'delivery', 'navigation']),
                        'success': np.random.random() > 0.2,  # 80% success rate
                        'execution_time': np.random.uniform(10, 30),
                        'position': agent.state.position,
                        'robot_state': {
                            'battery_level': agent.state.battery_level,
                            'task_load': agent.state.task_load
                        }
                    }
                    allocation_results.append(result)
                
                # Update Q-values
                coordinator.update_agent_q_values(allocation_results)
                
                # Track convergence
                convergence = coordinator.calculate_convergence()
                convergence_history.append(convergence)
            
            # Calculate final metrics
            final_convergence = convergence_history[-1] if convergence_history else 0.0
            avg_convergence = np.mean(convergence_history[-10:]) if len(convergence_history) >= 10 else 0.0
            
            results['metrics'] = {
                'episodes_run': num_episodes,
                'final_convergence': final_convergence,
                'avg_convergence_last_10': avg_convergence,
                'convergence_trend': np.polyfit(range(len(convergence_history)), convergence_history, 1)[0] if len(convergence_history) > 1 else 0.0
            }
            
            # Success criteria
            results['success'] = avg_convergence >= self.performance_targets['convergence_rate']
            
        except Exception as e:
            results['errors'].append(str(e))
            self.logger.error(f"Q-learning convergence test failed: {e}")
        
        return results
    
    async def test_auction_algorithm_performance(self) -> Dict:
        """Test auction algorithm performance"""
        self.logger.info("Testing auction algorithm performance")
        
        results = {
            'test_name': 'auction_performance',
            'success': False,
            'metrics': {},
            'errors': []
        }
        
        try:
            # Create auction allocator
            allocator = AuctionAllocator(auction_timeout=2.0)
            
            # Create test tasks and robot states
            tasks = []
            for i in range(10):
                task = self.task_generator.generate_task()
                tasks.append(task)
            
            robot_states = {}
            for agent in self.agents:
                robot_states[agent.robot_id] = agent.state
            
            # Run auctions
            allocation_times = []
            successful_allocations = 0
            
            for task in tasks:
                start_time = time.time()
                
                winner, bid = await allocator.run_auction(
                    task, 
                    list(robot_states.keys()), 
                    robot_states
                )
                
                allocation_time = (time.time() - start_time) * 1000  # ms
                allocation_times.append(allocation_time)
                
                if winner:
                    successful_allocations += 1
            
            # Calculate metrics
            avg_allocation_time = np.mean(allocation_times)
            success_rate = successful_allocations / len(tasks)
            
            # Get allocator statistics
            allocator_stats = allocator.get_auction_statistics()
            
            results['metrics'] = {
                'tasks_tested': len(tasks),
                'successful_allocations': successful_allocations,
                'success_rate': success_rate,
                'avg_allocation_time_ms': avg_allocation_time,
                'max_allocation_time_ms': max(allocation_times),
                'min_allocation_time_ms': min(allocation_times),
                'efficiency_improvement': allocator_stats.get('avg_efficiency_improvement', 0.0)
            }
            
            # Success criteria
            results['success'] = (
                success_rate >= 0.8 and 
                avg_allocation_time <= self.performance_targets['allocation_time']
            )
            
        except Exception as e:
            results['errors'].append(str(e))
            self.logger.error(f"Auction algorithm test failed: {e}")
        
        return results
    
    async def test_fault_tolerance(self) -> Dict:
        """Test fault tolerance and recovery"""
        self.logger.info("Testing fault tolerance")
        
        results = {
            'test_name': 'fault_tolerance',
            'success': False,
            'metrics': {},
            'errors': []
        }
        
        try:
            # Create fault tolerance manager
            ft_manager = FaultToleranceManager()
            
            # Simulate various faults
            fault_scenarios = [
                ('communication_failure', 'high'),
                ('battery_depletion', 'critical'),
                ('hardware_failure', 'medium'),
                ('task_timeout', 'medium')
            ]
            
            recovery_times = []
            successful_recoveries = 0
            
            for fault_type, severity in fault_scenarios:
                from communication.fault_tolerance import FaultType, FaultSeverity
                
                # Report fault
                fault_id = await ft_manager.report_fault(
                    FaultType(fault_type),
                    FaultSeverity(severity),
                    f"test_robot_{np.random.randint(0, self.num_robots)}",
                    f"Simulated {fault_type} fault"
                )
                
                # Wait for recovery
                start_time = time.time()
                timeout = 5.0  # 5 second timeout
                
                while fault_id in ft_manager.active_faults and time.time() - start_time < timeout:
                    await asyncio.sleep(0.1)
                
                recovery_time = time.time() - start_time
                recovery_times.append(recovery_time)
                
                if fault_id not in ft_manager.active_faults:
                    successful_recoveries += 1
            
            # Calculate metrics
            avg_recovery_time = np.mean(recovery_times)
            recovery_rate = successful_recoveries / len(fault_scenarios)
            
            # Get fault tolerance statistics
            ft_stats = ft_manager.get_fault_statistics()
            
            results['metrics'] = {
                'faults_simulated': len(fault_scenarios),
                'successful_recoveries': successful_recoveries,
                'recovery_rate': recovery_rate,
                'avg_recovery_time': avg_recovery_time,
                'max_recovery_time': max(recovery_times),
                'system_availability': ft_stats.get('system_availability', 0.0),
                'meets_failover_target': ft_stats.get('meets_failover_target', False)
            }
            
            # Success criteria
            results['success'] = (
                recovery_rate >= 0.8 and 
                avg_recovery_time <= 2.0  # 2 second target
            )
            
        except Exception as e:
            results['errors'].append(str(e))
            self.logger.error(f"Fault tolerance test failed: {e}")
        
        return results
    
    async def test_system_scalability(self) -> Dict:
        """Test system scalability with increasing robot count"""
        self.logger.info("Testing system scalability")
        
        results = {
            'test_name': 'system_scalability',
            'success': False,
            'metrics': {},
            'errors': []
        }
        
        try:
            # Test with increasing number of robots
            robot_counts = [5, 10, 15, 20]
            scalability_metrics = []
            
            for robot_count in robot_counts:
                # Create temporary agents
                temp_agents = []
                for i in range(robot_count):
                    agent = RobotAgent(f"scale_test_robot_{i}")
                    temp_agents.append(agent)
                
                # Register with master
                start_time = time.time()
                
                for agent in temp_agents:
                    await self.master.register_robot(
                        agent.robot_id,
                        agent.state.capabilities,
                        agent.state.position
                    )
                
                registration_time = time.time() - start_time
                
                # Generate tasks
                num_tasks = robot_count * 2  # 2 tasks per robot
                tasks = []
                
                task_start_time = time.time()
                for i in range(num_tasks):
                    task = self.task_generator.generate_task()
                    await self.master.submit_task(task)
                    tasks.append(task)
                
                task_generation_time = time.time() - task_start_time
                
                # Wait for allocation
                allocation_start_time = time.time()
                await asyncio.sleep(2.0)
                allocation_time = time.time() - allocation_start_time
                
                # Calculate metrics
                metrics = {
                    'robot_count': robot_count,
                    'registration_time': registration_time,
                    'task_generation_time': task_generation_time,
                    'allocation_time': allocation_time,
                    'tasks_per_second': num_tasks / task_generation_time,
                    'memory_usage': self.get_memory_usage()
                }
                
                scalability_metrics.append(metrics)
                
                # Clean up temporary agents
                for agent in temp_agents:
                    if agent.robot_id in self.master.robots:
                        del self.master.robots[agent.robot_id]
            
            # Analyze scalability
            max_robots_tested = max(robot_counts)
            final_metrics = scalability_metrics[-1]
            
            results['metrics'] = {
                'max_robots_tested': max_robots_tested,
                'scalability_data': scalability_metrics,
                'final_allocation_time': final_metrics['allocation_time'],
                'final_tasks_per_second': final_metrics['tasks_per_second'],
                'memory_growth': scalability_metrics[-1]['memory_usage'] - scalability_metrics[0]['memory_usage']
            }
            
            # Success criteria
            results['success'] = (
                max_robots_tested >= 20 and 
                final_metrics['allocation_time'] <= 5.0 and
                final_metrics['tasks_per_second'] >= 1.0
            )
            
        except Exception as e:
            results['errors'].append(str(e))
            self.logger.error(f"Scalability test failed: {e}")
        
        return results
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0  # Return 0 if psutil not available
    
    async def test_end_to_end_performance(self) -> Dict:
        """Test complete end-to-end system performance"""
        self.logger.info("Testing end-to-end performance")
        
        results = {
            'test_name': 'end_to_end_performance',
            'success': False,
            'metrics': {},
            'errors': []
        }
        
        try:
            # Start system components
            master_task = asyncio.create_task(self.run_master_loop())
            agent_tasks = [asyncio.create_task(self.run_agent_loop(agent)) for agent in self.agents]
            task_gen_task = asyncio.create_task(self.run_task_generation_loop())
            
            # Run for test duration
            await asyncio.sleep(self.test_duration)
            
            # Stop components
            master_task.cancel()
            for task in agent_tasks:
                task.cancel()
            task_gen_task.cancel()
            
            # Collect final metrics
            master_metrics = self.master.calculate_performance_metrics()
            task_gen_stats = self.task_generator.get_statistics()
            
            # Calculate system performance
            system_efficiency = master_metrics.get('efficiency', 0.0)
            avg_allocation_time = master_metrics.get('avg_allocation_time', 0.0)
            total_tasks = task_gen_stats.get('total_tasks_generated', 0)
            completed_tasks = master_metrics.get('tasks_completed', 0)
            
            results['metrics'] = {
                'test_duration': self.test_duration,
                'system_efficiency': system_efficiency,
                'avg_allocation_time_ms': avg_allocation_time,
                'total_tasks_generated': total_tasks,
                'tasks_completed': completed_tasks,
                'completion_rate': completed_tasks / max(total_tasks, 1),
                'tasks_per_second': total_tasks / self.test_duration,
                'convergence_rate': master_metrics.get('q_convergence', 0.0),
                'policy_gradient': master_metrics.get('policy_gradient', 0.0)
            }
            
            # Success criteria - must meet all performance targets
            success_criteria = [
                system_efficiency >= self.performance_targets['efficiency'],
                avg_allocation_time <= self.performance_targets['allocation_time'],
                results['metrics']['convergence_rate'] >= self.performance_targets['convergence_rate'],
                results['metrics']['completion_rate'] >= 0.8
            ]
            
            results['success'] = all(success_criteria)
            
        except Exception as e:
            results['errors'].append(str(e))
            self.logger.error(f"End-to-end performance test failed: {e}")
        
        return results
    
    async def run_master_loop(self):
        """Run coordination master loop"""
        try:
            # Simplified master loop for testing
            while True:
                await asyncio.sleep(0.1)
                
                # Process any pending tasks
                if self.master.task_queue:
                    await self.master.allocate_tasks()
                
        except asyncio.CancelledError:
            pass
    
    async def run_agent_loop(self, agent):
        """Run robot agent loop"""
        try:
            while True:
                # Simulate agent activity
                await asyncio.sleep(1.0)
                
                # Update agent state
                agent.state.battery_level = max(10, agent.state.battery_level - 0.5)
                
                # Simulate task completion
                if np.random.random() < 0.1:  # 10% chance per second
                    # Simulate completing a task
                    await self.master.handle_task_completion(
                        agent.robot_id, 
                        f"simulated_task_{int(time.time())}", 
                        True
                    )
                
        except asyncio.CancelledError:
            pass
    
    async def run_task_generation_loop(self):
        """Run task generation loop"""
        try:
            while True:
                # Generate task at specified rate
                interval = 1.0 / self.task_generation_rate
                
                task = self.task_generator.generate_task()
                await self.master.submit_task(task)
                
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            pass
    
    async def run_all_tests(self) -> Dict:
        """Run complete integration test suite"""
        self.logger.info("Starting complete integration test suite")
        
        # Setup test environment
        await self.setup_test_environment()
        
        # Define test sequence
        test_functions = [
            self.test_basic_coordination,
            self.test_q_learning_convergence,
            self.test_auction_algorithm_performance,
            self.test_fault_tolerance,
            self.test_system_scalability,
            self.test_end_to_end_performance
        ]
        
        # Run all tests
        test_results = []
        passed_tests = 0
        
        for test_func in test_functions:
            try:
                result = await test_func()
                test_results.append(result)
                
                if result['success']:
                    passed_tests += 1
                    self.logger.info(f"✓ {result['test_name']} PASSED")
                else:
                    self.logger.warning(f"✗ {result['test_name']} FAILED")
                    if result['errors']:
                        for error in result['errors']:
                            self.logger.error(f"  Error: {error}")
                
            except Exception as e:
                self.logger.error(f"Test {test_func.__name__} crashed: {e}")
                test_results.append({
                    'test_name': test_func.__name__,
                    'success': False,
                    'metrics': {},
                    'errors': [str(e)]
                })
        
        # Generate summary
        total_tests = len(test_functions)
        pass_rate = passed_tests / total_tests
        
        summary = {
            'test_suite': 'Multi-Robot Coordination Integration Tests',
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': pass_rate,
            'overall_success': pass_rate >= 0.8,  # 80% pass rate required
            'test_results': test_results,
            'performance_summary': self.generate_performance_summary(test_results)
        }
        
        # Log summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"INTEGRATION TEST SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Passed: {passed_tests}")
        self.logger.info(f"Failed: {total_tests - passed_tests}")
        self.logger.info(f"Pass Rate: {pass_rate:.1%}")
        self.logger.info(f"Overall Success: {'YES' if summary['overall_success'] else 'NO'}")
        
        return summary
    
    def generate_performance_summary(self, test_results: List[Dict]) -> Dict:
        """Generate performance summary from test results"""
        performance_summary = {
            'coordination_efficiency': 0.0,
            'allocation_performance': 0.0,
            'learning_convergence': 0.0,
            'fault_tolerance_score': 0.0,
            'scalability_score': 0.0,
            'overall_score': 0.0
        }
        
        try:
            # Extract key performance metrics
            for result in test_results:
                if not result['success']:
                    continue
                
                metrics = result['metrics']
                test_name = result['test_name']
                
                if test_name == 'basic_coordination':
                    performance_summary['coordination_efficiency'] = metrics.get('allocation_rate', 0.0)
                
                elif test_name == 'auction_performance':
                    # Normalize allocation time performance (lower is better)
                    alloc_time = metrics.get('avg_allocation_time_ms', 100.0)
                    performance_summary['allocation_performance'] = min(1.0, 50.0 / alloc_time)
                
                elif test_name == 'q_learning_convergence':
                    performance_summary['learning_convergence'] = metrics.get('avg_convergence_last_10', 0.0)
                
                elif test_name == 'fault_tolerance':
                    performance_summary['fault_tolerance_score'] = metrics.get('recovery_rate', 0.0)
                
                elif test_name == 'system_scalability':
                    # Score based on maximum robots tested
                    max_robots = metrics.get('max_robots_tested', 0)
                    performance_summary['scalability_score'] = min(1.0, max_robots / 20.0)
            
            # Calculate overall score
            scores = [score for score in performance_summary.values() if score > 0]
            performance_summary['overall_score'] = np.mean(scores) if scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
        
        return performance_summary


class TestRunner:
    """Test runner for integration tests"""
    
    @staticmethod
    async def run_quick_test():
        """Run quick integration test"""
        test_suite = IntegrationTestSuite()
        test_suite.num_robots = 2
        test_suite.test_duration = 10.0
        
        return await test_suite.run_all_tests()
    
    @staticmethod
    async def run_full_test():
        """Run full integration test"""
        test_suite = IntegrationTestSuite()
        return await test_suite.run_all_tests()
    
    @staticmethod
    async def run_performance_benchmark():
        """Run performance benchmark"""
        test_suite = IntegrationTestSuite()
        test_suite.num_robots = 10
        test_suite.test_duration = 60.0
        test_suite.task_generation_rate = 2.0
        
        return await test_suite.run_all_tests()


async def main():
    """Main entry point for integration tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Robot Integration Tests")
    parser.add_argument("--test-type", choices=["quick", "full", "benchmark"], 
                       default="quick", help="Type of test to run")
    parser.add_argument("--robots", type=int, default=3, help="Number of robots")
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration (seconds)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    if args.test_type == "quick":
        results = await TestRunner.run_quick_test()
    elif args.test_type == "full":
        results = await TestRunner.run_full_test()
    elif args.test_type == "benchmark":
        results = await TestRunner.run_performance_benchmark()
    else:
        # Custom test
        test_suite = IntegrationTestSuite()
        test_suite.num_robots = args.robots
        test_suite.test_duration = args.duration
        results = await test_suite.run_all_tests()
    
    # Print results
    print("\n" + "="*80)
    print("INTEGRATION TEST RESULTS")
    print("="*80)
    
    print(f"Overall Success: {'PASS' if results['overall_success'] else 'FAIL'}")
    print(f"Pass Rate: {results['pass_rate']:.1%}")
    print(f"Tests Passed: {results['passed_tests']}/{results['total_tests']}")
    
    print("\nPerformance Summary:")
    perf = results['performance_summary']
    print(f"  Overall Score: {perf['overall_score']:.3f}")
    print(f"  Coordination Efficiency: {perf['coordination_efficiency']:.3f}")
    print(f"  Allocation Performance: {perf['allocation_performance']:.3f}")
    print(f"  Learning Convergence: {perf['learning_convergence']:.3f}")
    print(f"  Fault Tolerance: {perf['fault_tolerance_score']:.3f}")
    print(f"  Scalability: {perf['scalability_score']:.3f}")
    
    print("\nIndividual Test Results:")
    for test_result in results['test_results']:
        status = "PASS" if test_result['success'] else "FAIL"
        print(f"  {test_result['test_name']}: {status}")
        if test_result['errors']:
            for error in test_result['errors']:
                print(f"    Error: {error}")
    
    # Export results
    import json
    with open(f"integration_test_results_{int(time.time())}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results exported to: integration_test_results_{int(time.time())}.json")
    
    # Exit with appropriate code
    exit(0 if results['overall_success'] else 1)


if __name__ == "__main__":
    asyncio.run(main())
