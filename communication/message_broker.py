
#!/usr/bin/env python3
"""
Message Broker for Multi-Robot Coordination Framework
Handles message routing, queuing, and reliability
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import threading
import queue

@dataclass
class Message:
    """Message structure for inter-robot communication"""
    sender_id: str
    receiver_id: str
    message_type: str
    timestamp: float
    data: Dict[str, Any]
    message_id: str
    priority: int = 1
    ttl: float = 30.0  # Time to live in seconds
    retry_count: int = 0
    max_retries: int = 3

class MessageBroker:
    """Central message broker for reliable communication"""
    
    def __init__(self):
        # Message queues
        self.message_queues: Dict[str, queue.Queue] = defaultdict(lambda: queue.Queue())
        self.pending_messages: Dict[str, Message] = {}
        self.delivered_messages: Dict[str, float] = {}  # message_id -> delivery_time
        
        # Routing table
        self.routing_table: Dict[str, str] = {}  # robot_id -> endpoint
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Performance metrics
        self.messages_sent = 0
        self.messages_delivered = 0
        self.messages_failed = 0
        self.latency_history = deque(maxlen=1000)
        
        # Configuration
        self.cleanup_interval = 30.0  # seconds
        self.max_queue_size = 1000
        
        self.logger = logging.getLogger("message_broker")
        self.logger.info("Message broker initialized")
        
        # Start background tasks
        self._running = True
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup of expired messages"""
        while self._running:
            try:
                current_time = time.time()
                expired_messages = []
                
                # Find expired messages
                for msg_id, message in self.pending_messages.items():
                    if current_time - message.timestamp > message.ttl:
                        expired_messages.append(msg_id)
                
                # Remove expired messages
                for msg_id in expired_messages:
                    message = self.pending_messages.pop(msg_id, None)
                    if message:
                        self.messages_failed += 1
                        self.logger.warning(f"Message {msg_id} expired after {message.ttl}s")
                
                # Clean up old delivery records
                old_deliveries = [
                    msg_id for msg_id, delivery_time in self.delivered_messages.items()
                    if current_time - delivery_time > 300  # 5 minutes
                ]
                
                for msg_id in old_deliveries:
                    del self.delivered_messages[msg_id]
                
                time.sleep(self.cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                time.sleep(5.0)
    
    async def send_message(self, message: Message) -> bool:
        """Send message with reliability guarantees"""
        try:
            # Validate message
            if not self._validate_message(message):
                return False
            
            # Add to pending messages
            self.pending_messages[message.message_id] = message
            self.messages_sent += 1
            
            # Route message
            success = await self._route_message(message)
            
            if success:
                # Mark as delivered
                self.delivered_messages[message.message_id] = time.time()
                self.messages_delivered += 1
                
                # Calculate latency
                latency = (time.time() - message.timestamp) * 1000  # ms
                self.latency_history.append(latency)
                
                # Remove from pending
                self.pending_messages.pop(message.message_id, None)
                
                self.logger.debug(f"Message {message.message_id} delivered successfully")
            else:
                # Handle retry
                await self._handle_retry(message)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending message {message.message_id}: {e}")
            return False
    
    def _validate_message(self, message: Message) -> bool:
        """Validate message structure and content"""
        if not message.sender_id or not message.receiver_id:
            self.logger.error("Message missing sender or receiver ID")
            return False
        
        if not message.message_type:
            self.logger.error("Message missing type")
            return False
        
        if not message.data:
            self.logger.warning("Message has empty data")
        
        return True
    
    async def _route_message(self, message: Message) -> bool:
        """Route message to appropriate destination"""
        receiver_id = message.receiver_id
        
        # Broadcast message
        if receiver_id == "all" or receiver_id == "*":
            return await self._broadcast_message(message)
        
        # Direct message
        if receiver_id in self.routing_table:
            endpoint = self.routing_table[receiver_id]
            return await self._deliver_to_endpoint(message, endpoint)
        
        # Queue for later delivery
        if receiver_id not in self.message_queues:
            self.message_queues[receiver_id] = queue.Queue(maxsize=self.max_queue_size)
        
        try:
            self.message_queues[receiver_id].put_nowait(message)
            self.logger.debug(f"Message queued for {receiver_id}")
            return True
        except queue.Full:
            self.logger.error(f"Message queue full for {receiver_id}")
            return False
    
    async def _broadcast_message(self, message: Message) -> bool:
        """Broadcast message to all registered endpoints"""
        success_count = 0
        total_endpoints = len(self.routing_table)
        
        if total_endpoints == 0:
            self.logger.warning("No endpoints registered for broadcast")
            return False
        
        for robot_id, endpoint in self.routing_table.items():
            if robot_id != message.sender_id:  # Don't send to sender
                success = await self._deliver_to_endpoint(message, endpoint)
                if success:
                    success_count += 1
        
        # Consider broadcast successful if majority delivered
        return success_count >= (total_endpoints * 0.5)
    
    async def _deliver_to_endpoint(self, message: Message, endpoint: str) -> bool:
        """Deliver message to specific endpoint"""
        try:
            # In a real implementation, this would use the actual transport
            # For simulation, we'll just trigger callbacks
            
            # Notify subscribers
            if message.message_type in self.subscribers:
                for callback in self.subscribers[message.message_type]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        self.logger.error(f"Subscriber callback error: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Delivery error to {endpoint}: {e}")
            return False
    
    async def _handle_retry(self, message: Message):
        """Handle message retry logic"""
        message.retry_count += 1
        
        if message.retry_count <= message.max_retries:
            # Exponential backoff
            retry_delay = 2 ** message.retry_count
            self.logger.info(f"Retrying message {message.message_id} in {retry_delay}s (attempt {message.retry_count})")
            
            # Schedule retry
            asyncio.create_task(self._delayed_retry(message, retry_delay))
        else:
            # Max retries exceeded
            self.messages_failed += 1
            self.pending_messages.pop(message.message_id, None)
            self.logger.error(f"Message {message.message_id} failed after {message.max_retries} retries")
    
    async def _delayed_retry(self, message: Message, delay: float):
        """Retry message after delay"""
        await asyncio.sleep(delay)
        await self.send_message(message)
    
    def register_endpoint(self, robot_id: str, endpoint: str):
        """Register robot endpoint for message routing"""
        self.routing_table[robot_id] = endpoint
        self.logger.info(f"Registered endpoint for {robot_id}: {endpoint}")
        
        # Deliver any queued messages
        if robot_id in self.message_queues:
            asyncio.create_task(self._deliver_queued_messages(robot_id))
    
    async def _deliver_queued_messages(self, robot_id: str):
        """Deliver any queued messages for a robot"""
        message_queue = self.message_queues[robot_id]
        endpoint = self.routing_table[robot_id]
        
        delivered_count = 0
        while not message_queue.empty():
            try:
                message = message_queue.get_nowait()
                
                # Check if message is still valid
                if time.time() - message.timestamp > message.ttl:
                    continue
                
                success = await self._deliver_to_endpoint(message, endpoint)
                if success:
                    delivered_count += 1
                    self.delivered_messages[message.message_id] = time.time()
                    self.messages_delivered += 1
                else:
                    # Put back in queue for retry
                    message_queue.put_nowait(message)
                    break
                
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error delivering queued message: {e}")
        
        if delivered_count > 0:
            self.logger.info(f"Delivered {delivered_count} queued messages to {robot_id}")
    
    def unregister_endpoint(self, robot_id: str):
        """Unregister robot endpoint"""
        if robot_id in self.routing_table:
            del self.routing_table[robot_id]
            self.logger.info(f"Unregistered endpoint for {robot_id}")
    
    def subscribe(self, message_type: str, callback: Callable):
        """Subscribe to specific message types"""
        self.subscribers[message_type].append(callback)
        self.logger.debug(f"Subscribed to message type: {message_type}")
    
    def unsubscribe(self, message_type: str, callback: Callable):
        """Unsubscribe from message type"""
        if callback in self.subscribers[message_type]:
            self.subscribers[message_type].remove(callback)
            self.logger.debug(f"Unsubscribed from message type: {message_type}")
    
    async def receive_message(self, robot_id: str, timeout: float = 1.0) -> Optional[Message]:
        """Receive message for specific robot"""
        if robot_id not in self.message_queues:
            return None
        
        message_queue = self.message_queues[robot_id]
        
        try:
            # Non-blocking get with timeout simulation
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    message = message_queue.get_nowait()
                    
                    # Check if message is still valid
                    if time.time() - message.timestamp <= message.ttl:
                        return message
                    
                except queue.Empty:
                    await asyncio.sleep(0.1)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error receiving message for {robot_id}: {e}")
            return None
    
    async def check_communication_health(self) -> List[str]:
        """Check communication system health"""
        issues = []
        
        # Check queue sizes
        for robot_id, message_queue in self.message_queues.items():
            queue_size = message_queue.qsize()
            if queue_size > self.max_queue_size * 0.8:
                issues.append(f"High queue size for {robot_id}: {queue_size}")
        
        # Check failure rate
        total_messages = self.messages_sent
        if total_messages > 0:
            failure_rate = self.messages_failed / total_messages
            if failure_rate > 0.05:  # 5% failure rate threshold
                issues.append(f"High message failure rate: {failure_rate:.2%}")
        
        # Check latency
        if self.latency_history:
            avg_latency = sum(self.latency_history) / len(self.latency_history)
            if avg_latency > 100.0:  # 100ms threshold
                issues.append(f"High average latency: {avg_latency:.1f}ms")
        
        # Check pending messages
        pending_count = len(self.pending_messages)
        if pending_count > 100:
            issues.append(f"High pending message count: {pending_count}")
        
        return issues
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get message broker statistics"""
        avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0.0
        
        return {
            'messages_sent': self.messages_sent,
            'messages_delivered': self.messages_delivered,
            'messages_failed': self.messages_failed,
            'pending_messages': len(self.pending_messages),
            'registered_endpoints': len(self.routing_table),
            'active_queues': len(self.message_queues),
            'avg_latency_ms': avg_latency,
            'delivery_rate': self.messages_delivered / max(self.messages_sent, 1),
            'failure_rate': self.messages_failed / max(self.messages_sent, 1)
        }
    
    def shutdown(self):
        """Shutdown message broker"""
        self._running = False
        self.logger.info("Message broker shutting down")


if __name__ == "__main__":
    # Test the message broker
    async def test_message_broker():
        broker = MessageBroker()
        
        # Register some endpoints
        broker.register_endpoint("robot_1", "endpoint_1")
        broker.register_endpoint("robot_2", "endpoint_2")
        
        # Create test message
        test_message = Message(
            sender_id="test_sender",
            receiver_id="robot_1",
            message_type="test_message",
            timestamp=time.time(),
            data={"test": "data"},
            message_id="test_001"
        )
        
        # Send message
        success = await broker.send_message(test_message)
        print(f"Message sent: {success}")
        
        # Get statistics
        stats = broker.get_statistics()
        print(f"Broker statistics: {stats}")
        
        broker.shutdown()
    
    asyncio.run(test_message_broker())
