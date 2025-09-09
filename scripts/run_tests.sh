#!/bin/bash
"""
Test runner script for Multi-Robot Coordination Framework
Runs unit tests, integration tests, and performance benchmarks
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FRAMEWORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="${FRAMEWORK_DIR}/src/tests"
LOG_DIR="${FRAMEWORK_DIR}/logs"
RESULTS_DIR="${FRAMEWORK_DIR}/test_results"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

echo -e "${BLUE}Multi-Robot Coordination Framework Test Suite${NC}"
echo "=============================================="
echo "Framework Directory: ${FRAMEWORK_DIR}"
echo "Test Directory: ${TEST_DIR}"
echo ""

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
    esac
}

# Function to check dependencies
check_dependencies() {
    print_status "INFO" "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_status "ERROR" "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check pip packages
    local required_packages=("pytest" "numpy" "pyyaml" "asyncio")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import ${package}" &> /dev/null; then
            print_status "WARNING" "Package ${package} not found, installing..."
            pip3 install ${package}
        fi
    done
    
    print_status "SUCCESS" "Dependencies check completed"
}

# Function to run unit tests
run_unit_tests() {
    print_status "INFO" "Running unit tests..."
    
    cd "${FRAMEWORK_DIR}"
    
    # Create unit test files if they don't exist
    if [ ! -f "${TEST_DIR}/test_coordination.py" ]; then
        cat > "${TEST_DIR}/test_coordination.py" << 'EOF'
#!/usr/bin/env python3
import pytest
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from coordination_master import CoordinationMaster
from robot_agent import RobotAgent

class TestCoordination:
    @pytest.mark.asyncio
    async def test_robot_registration(self):
        master = CoordinationMaster()
        success = await master.register_robot("test_robot", ["navigation"], (0, 0))
        assert success == True
        assert "test_robot" in master.robots

    def test_task_creation(self):
        from coordination_master import Task
        task = Task(
            task_id="test_001",
            task_type="navigation",
            priority=1.0,
            location=(10, 20),
            deadline=1000.0,
            required_capabilities=["navigation"],
            estimated_duration=30.0
        )
        assert task.task_id == "test_001"
        assert task.task_type == "navigation"

if __name__ == "__main__":
    pytest.main([__file__])
EOF
    fi
    
    if [ ! -f "${TEST_DIR}/test_algorithms.py" ]; then
        cat > "${TEST_DIR}/test_algorithms.py" << 'EOF'
#!/usr/bin/env python3
import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.q_learning import QLearningAgent, QLearningCoordinator
from algorithms.auction_algorithm import AuctionAllocator

class TestQLearning:
    def test_q_learning_agent_creation(self):
        agent = QLearningAgent("test_agent")
        assert agent.agent_id == "test_agent"
        assert agent.learning_rate > 0
        assert agent.exploration_rate > 0

    def test_q_value_update(self):
        agent = QLearningAgent("test_agent")
        state = ("navigation", 100, 0, 0, 0)
        action = "execute"
        reward = 1.0
        
        initial_q = agent.get_q_value(state, action)
        agent.update_q_value(state, action, reward)
        updated_q = agent.get_q_value(state, action)
        
        assert updated_q != initial_q

class TestAuction:
    @pytest.mark.asyncio
    async def test_auction_creation(self):
        allocator = AuctionAllocator()
        assert allocator.auction_timeout > 0
        assert allocator.min_bid_threshold >= 0

if __name__ == "__main__":
    pytest.main([__file__])
EOF
    fi
    
    # Run pytest
    python3 -m pytest "${TEST_DIR}" -v --tb=short --color=yes \
        --junitxml="${RESULTS_DIR}/unit_test_results.xml" \
        2>&1 | tee "${LOG_DIR}/unit_tests.log"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        print_status "SUCCESS" "Unit tests passed"
        return 0
    else
        print_status "ERROR" "Unit tests failed"
        return 1
    fi
}

# Function to run integration tests
run_integration_tests() {
    print_status "INFO" "Running integration tests..."
    
    cd "${FRAMEWORK_DIR}"
    
    # Run integration tests
    python3 "${TEST_DIR}/integration_test.py" --test-type quick --verbose \
        2>&1 | tee "${LOG_DIR}/integration_tests.log"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        print_status "SUCCESS" "Integration tests passed"
        return 0
    else
        print_status "ERROR" "Integration tests failed"
        return 1
    fi
}

# Function to run performance benchmarks
run_performance_tests() {
    print_status "INFO" "Running performance benchmarks..."
    
    cd "${FRAMEWORK_DIR}"
    
    # Run performance benchmarks
    python3 "${TEST_DIR}/integration_test.py" --test-type benchmark \
        --robots 10 --duration 60 --verbose \
        2>&1 | tee "${LOG_DIR}/performance_tests.log"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        print_status "SUCCESS" "Performance tests passed"
        return 0
    else
        print_status "ERROR" "Performance tests failed"
        return 1
    fi
}

# Function to run system validation
run_system_validation() {
    print_status "INFO" "Running system validation..."
    
    # Create validation script
    cat > "${RESULTS_DIR}/system_validation.py" << 'EOF'
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def validate_system():
    """Validate system components and configuration"""
    results = {
        'imports': True,
        'config': True,
        'structure': True
    }
    
    # Test imports
    try:
        from coordination_master import CoordinationMaster
        from robot_agent import RobotAgent
        from task_generator import TaskGenerator
        from algorithms.q_learning import QLearningCoordinator
        from algorithms.auction_algorithm import AuctionAllocator
        print("✓ All core modules import successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        results['imports'] = False
    
    # Test configuration
    try:
        from utils.config import ConfigManager
        config = ConfigManager("config/system_config.yaml")
        assert config.get("coordination.max_robots", 0) > 0
        print("✓ Configuration system working")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        results['config'] = False
    
    # Test directory structure
    required_dirs = ['src', 'config', 'logs']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"✗ Missing directory: {dir_name}")
            results['structure'] = False
        else:
            print(f"✓ Directory exists: {dir_name}")
    
    # Overall result
    all_passed = all(results.values())
    print(f"\nSystem validation: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed

if __name__ == "__main__":
    success = validate_system()
    sys.exit(0 if success else 1)
EOF
    
    python3 "${RESULTS_DIR}/system_validation.py" 2>&1 | tee "${LOG_DIR}/system_validation.log"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        print_status "SUCCESS" "System validation passed"
        return 0
    else
        print_status "ERROR" "System validation failed"
        return 1
    fi
}

# Function to generate test report
generate_test_report() {
    print_status "INFO" "Generating test report..."
    
    local timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
    local report_file="${RESULTS_DIR}/test_report_${timestamp}.html"
    
    cat > "${report_file}" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Robot Coordination Framework Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
        .info { background-color: #d1ecf1; color: #0c5460; }
        pre { background-color: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Multi-Robot Coordination Framework Test Report</h1>
        <p>Generated: $(date)</p>
        <p>Framework Directory: ${FRAMEWORK_DIR}</p>
    </div>
    
    <div class="section info">
        <h2>Test Summary</h2>
        <ul>
            <li>Unit Tests: $([ -f "${LOG_DIR}/unit_tests.log" ] && echo "Completed" || echo "Not Run")</li>
            <li>Integration Tests: $([ -f "${LOG_DIR}/integration_tests.log" ] && echo "Completed" || echo "Not Run")</li>
            <li>Performance Tests: $([ -f "${LOG_DIR}/performance_tests.log" ] && echo "Completed" || echo "Not Run")</li>
            <li>System Validation: $([ -f "${LOG_DIR}/system_validation.log" ] && echo "Completed" || echo "Not Run")</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Test Logs</h2>
        <p>Detailed logs are available in the <code>${LOG_DIR}</code> directory:</p>
        <ul>
EOF
    
    # Add log files to report
    for log_file in "${LOG_DIR}"/*.log; do
        if [ -f "$log_file" ]; then
            echo "            <li><a href=\"file://${log_file}\">$(basename "$log_file")</a></li>" >> "${report_file}"
        fi
    done
    
    cat >> "${report_file}" << EOF
        </ul>
    </div>
    
    <div class="section">
        <h2>Performance Metrics</h2>
        <p>Performance test results and metrics are available in the test result files.</p>
    </div>
    
    <div class="section">
        <h2>Framework Structure</h2>
        <pre>
$(tree "${FRAMEWORK_DIR}" -I '__pycache__|*.pyc|.git' 2>/dev/null || find "${FRAMEWORK_DIR}" -type f -name "*.py" | head -20)
        </pre>
    </div>
    
</body>
</html>
EOF
    
    print_status "SUCCESS" "Test report generated: ${report_file}"
}

# Main execution
main() {
    local run_unit=true
    local run_integration=true
    local run_performance=false
    local run_validation=true
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --unit-only)
                run_integration=false
                run_performance=false
                run_validation=false
                shift
                ;;
            --integration-only)
                run_unit=false
                run_performance=false
                run_validation=false
                shift
                ;;
            --performance)
                run_performance=true
                shift
                ;;
            --no-validation)
                run_validation=false
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --unit-only       Run only unit tests"
                echo "  --integration-only Run only integration tests"
                echo "  --performance     Include performance benchmarks"
                echo "  --no-validation   Skip system validation"
                echo "  --help           Show this help message"
                exit 0
                ;;
            *)
                print_status "ERROR" "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Start testing
    print_status "INFO" "Starting test execution..."
    
    local overall_success=true
    
    # Check dependencies
    check_dependencies || overall_success=false
    
    # Run system validation
    if [ "$run_validation" = true ]; then
        run_system_validation || overall_success=false
    fi
    
    # Run unit tests
    if [ "$run_unit" = true ]; then
        run_unit_tests || overall_success=false
    fi
    
    # Run integration tests
    if [ "$run_integration" = true ]; then
        run_integration_tests || overall_success=false
    fi
    
    # Run performance tests
    if [ "$run_performance" = true ]; then
        run_performance_tests || overall_success=false
    fi
    
    # Generate report
    generate_test_report
    
    # Final summary
    echo ""
    echo "=============================================="
    if [ "$overall_success" = true ]; then
        print_status "SUCCESS" "All tests completed successfully!"
        echo -e "${GREEN}✓ Framework is ready for deployment${NC}"
    else
        print_status "ERROR" "Some tests failed!"
        echo -e "${RED}✗ Please check the logs and fix issues before deployment${NC}"
    fi
    echo "=============================================="
    
    # Exit with appropriate code
    if [ "$overall_success" = true ]; then
        exit 0
    else
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
