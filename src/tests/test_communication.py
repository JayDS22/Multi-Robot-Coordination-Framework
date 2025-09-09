#!/usr/bin/env python3
"""
Unit tests for communication functionality
"""

import pytest
import asyncio
import time
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from communication.ros_interface import ROSInterface, Message
from communication.fault_tolerance import FaultToleranceManager, FaultType, FaultSeverity, Fault
from communication.message_broker import MessageBroker

class TestROSInterface:
    """Test cases for ROS Interface"""
    
    @pytest.fixture
    async def ros_interface(self):
        """Create a test ROS interface"""
        interfa
