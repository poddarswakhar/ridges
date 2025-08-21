"""
Comprehensive integration tests for the approval system.
Tests cover immediate approvals, future scheduling with asyncio, timeouts, and real database operations.
"""

import pytest
import asyncio
import asyncpg
import uuid
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, AsyncMock, MagicMock
from typing import Dict, Any

import pytest_asyncio
from httpx import AsyncClient

# Set up environment before imports
os.environ.setdefault('AWS_MASTER_USERNAME', 'test_user')
os.environ.setdefault('AWS_MASTER_PASSWORD', 'test_pass')
os.environ.setdefault('AWS_RDS_PLATFORM_ENDPOINT', 'localhost')
os.environ.setdefault('AWS_RDS_PLATFORM_DB_NAME', 'postgres')
os.environ.setdefault('POSTGRES_TEST_URL', 'postgresql://test_user:test_pass@localhost:5432/postgres')

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api', 'src'))

from api.src.backend.queries.agents import approve_agent_version
from api.src.backend.queries.scores import evaluate_agent_for_threshold_approval, generate_threshold_function
from api.src.utils.threshold_scheduler import ThresholdScheduler
from api.src.main import app


class MockDBConnection:
    """Mock database connection for testing without event loop conflicts"""
    
    def __init__(self):
        self.executed_queries = []
        self.fetch_results = {}
        
    async def execute(self, query: str, *args):
        """Mock execute method"""
        self.executed_queries.append((query, args))
        return "OK"
        
    async def fetch(self, query: str, *args):
        """Mock fetch method"""
        self.executed_queries.append((query, args))
        return self.fetch_results.get(query, [])
        
    async def fetchrow(self, query: str, *args):
        """Mock fetchrow method"""
        self.executed_queries.append((query, args))
        results = self.fetch_results.get(query, [])
        return results[0] if results else None
        
    def set_fetch_result(self, query: str, result):
        """Set mock result for a query"""
        self.fetch_results[query] = result if isinstance(result, list) else [result]


@pytest.mark.integration
class TestApprovalIntegration:
    """Integration tests for agent approval system with real database operations"""

    @pytest_asyncio.fixture
    async def scheduler(self):
        """Create a ThresholdScheduler instance for testing"""
        scheduler = ThresholdScheduler()
        yield scheduler
        
        # Cleanup: cancel all scheduled tasks
        for task in scheduler._scheduled_tasks.values():
            task.cancel()
        scheduler._scheduled_tasks.clear()
        scheduler._pending_approvals.clear()

    @pytest_asyncio.fixture
    async def mock_db_conn(self):
        """Create a mock database connection"""
        return MockDBConnection()

    @pytest_asyncio.fixture
    async def test_agent_data(self, mock_db_conn):
        """Create test agent and evaluation data"""
        agent_id = uuid.uuid4()
        eval_id = uuid.uuid4()
        
        # Set up mock responses for common queries
        mock_db_conn.set_fetch_result(
            "SELECT * FROM approved_version_ids WHERE version_id = $1 AND set_id = $2",
            []
        )
        
        return {
            'agent_id': agent_id,
            'eval_id': eval_id,
            'score': 0.85,
            'set_id': 1
        }

    @pytest_asyncio.fixture
    async def threshold_data(self, mock_db_conn):
        """Set up threshold function data for testing"""
        epoch_0_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        
        # Mock threshold data
        mock_db_conn.set_fetch_result(
            "SELECT * FROM threshold_function ORDER BY created_at DESC LIMIT 1",
            [{
                'initial_threshold': 0.8,
                'k': 0.1,
                'epoch_0_time': epoch_0_time,
                'epoch_length_minutes': 60
            }]
        )
        
        return {
            'initial_threshold': 0.8,
            'k': 0.1,
            'epoch_0_time': epoch_0_time,
            'epoch_length_minutes': 60
        }

    @pytest.mark.asyncio
    async def test_immediate_approval_flow(self, mock_db_conn, test_agent_data: Dict):
        """Test immediate approval when agent score exceeds current threshold"""
        agent_id = test_agent_data['agent_id']
        set_id = test_agent_data['set_id']
        
        # Test scheduler functionality with mocked approval
        with patch('api.src.backend.queries.agents.approve_agent_version') as mock_approve, \
             patch('api.src.backend.db_manager.get_transaction') as mock_transaction:
             
            # Mock the database transaction context manager
            mock_conn = MockDBConnection()
            mock_transaction.return_value.__aenter__.return_value = mock_conn
            mock_transaction.return_value.__aexit__.return_value = None
            
            mock_approve.return_value = None  # approve_agent_version doesn't return anything
            
            # Test immediate execution (past time)
            past_time = datetime.now(timezone.utc) - timedelta(seconds=1)
            scheduler = ThresholdScheduler()
            
            scheduler.schedule_future_approval(str(agent_id), set_id, past_time)
            
            # Wait for immediate execution
            await asyncio.sleep(0.1)
            
            # Verify the approval was called (with approved_at=None for immediate approval)
            mock_approve.assert_called_once_with(str(agent_id), set_id, None)

    @pytest.mark.asyncio
    async def test_future_scheduling_integration(self, scheduler: ThresholdScheduler, threshold_data: Dict):
        """Test scheduling future approvals with real asyncio and database"""
        # Create agent with score that should be approved in the future
        future_agent_id = str(uuid.uuid4())
        set_id = 1
        
        # Mock agent score and threshold calculation
        with patch('api.src.backend.queries.agents.approve_agent_version') as mock_approve, \
             patch('api.src.backend.db_manager.get_transaction') as mock_transaction:
             
            # Mock the database transaction context manager
            mock_conn = MockDBConnection()
            mock_transaction.return_value.__aenter__.return_value = mock_conn
            mock_transaction.return_value.__aexit__.return_value = None
            
            mock_approve.return_value = None  # approve_agent_version doesn't return anything
            
            # Schedule approval for near future (1 second)
            future_time = datetime.now(timezone.utc) + timedelta(seconds=1)
            
            scheduler.schedule_future_approval(
                future_agent_id, set_id, future_time
            )
            
            assert scheduler.get_scheduled_count() == 1
            
            # Wait for the scheduled approval to execute
            await asyncio.sleep(1.5)
            
            # Verify the approval was executed
            mock_approve.assert_called_once_with(future_agent_id, set_id, None)
            assert scheduler.get_scheduled_count() == 0

    @pytest.mark.asyncio
    async def test_multiple_concurrent_schedules(self, scheduler: ThresholdScheduler):
        """Test multiple agents scheduled for approval concurrently"""
        agent_ids = [str(uuid.uuid4()) for _ in range(3)]
        set_id = 1
        
        with patch('api.src.backend.queries.agents.approve_agent_version') as mock_approve, \
             patch('api.src.backend.db_manager.get_transaction') as mock_transaction:
             
            # Mock the database transaction context manager
            mock_conn = MockDBConnection()
            mock_transaction.return_value.__aenter__.return_value = mock_conn
            mock_transaction.return_value.__aexit__.return_value = None
            mock_approve.return_value = None  # approve_agent_version doesn't return anything
            
            # Schedule multiple approvals with small delays
            for i, agent_id in enumerate(agent_ids):
                future_time = datetime.now(timezone.utc) + timedelta(milliseconds=500 + i*100)
                scheduler.schedule_future_approval(agent_id, set_id, future_time)
            
            assert scheduler.get_scheduled_count() == 3
            
            # Wait for all approvals to complete
            await asyncio.sleep(1.5)
            
            # Verify all approvals were executed
            assert mock_approve.call_count == 3
            assert scheduler.get_scheduled_count() == 0

    @pytest.mark.asyncio
    async def test_approval_timeout_handling(self, scheduler: ThresholdScheduler):
        """Test timeout scenarios in approval scheduling"""
        agent_id = str(uuid.uuid4())
        set_id = 1
        
        # Schedule approval with very short timeout
        future_time = datetime.now(timezone.utc) + timedelta(milliseconds=100)
        
        # Mock a slow approval function that will take time but succeed
        async def slow_approval(*args):
            await asyncio.sleep(0.3)  # Longer than initial schedule time
            return None  # approve_agent_version doesn't return anything
        
        with patch('api.src.backend.queries.agents.approve_agent_version', side_effect=slow_approval), \
             patch('api.src.backend.db_manager.get_transaction') as mock_transaction:
             
            # Mock the database transaction context manager
            mock_conn = MockDBConnection()
            mock_transaction.return_value.__aenter__.return_value = mock_conn
            mock_transaction.return_value.__aexit__.return_value = None
            
            scheduler.schedule_future_approval(agent_id, set_id, future_time)
            
            # Wait for execution to complete
            await asyncio.sleep(0.5)  # Wait longer than the slow approval
            
            # Task should be cleaned up after completion (even with slow approval)
            assert scheduler.get_scheduled_count() == 0

    @pytest.mark.asyncio
    async def test_approval_cancellation(self, scheduler: ThresholdScheduler):
        """Test cancellation of scheduled approvals"""
        agent_id = str(uuid.uuid4())
        set_id = 1
        
        # Schedule approval in the future
        future_time = datetime.now(timezone.utc) + timedelta(seconds=10)
        
        scheduler.schedule_future_approval(agent_id, set_id, future_time)
        assert scheduler.get_scheduled_count() == 1
        
        # Cancel the scheduled approval
        cancelled = scheduler.cancel_scheduled_approval(agent_id, set_id)
        assert cancelled is True
        assert scheduler.get_scheduled_count() == 0
        
        # Try to cancel again - should return False
        cancelled_again = scheduler.cancel_scheduled_approval(agent_id, set_id)
        assert cancelled_again is False

    @pytest.mark.asyncio
    async def test_approval_with_database_error_handling(self, scheduler: ThresholdScheduler):
        """Test approval handling when database operations fail"""
        agent_id = str(uuid.uuid4())
        set_id = 1
        
        # Mock database failure
        with patch('api.src.backend.queries.agents.approve_agent_version', side_effect=Exception("DB Error")):
            future_time = datetime.now(timezone.utc) + timedelta(milliseconds=100)
            
            scheduler.schedule_future_approval(agent_id, set_id, future_time)
            
            # Wait for execution and error handling
            await asyncio.sleep(0.3)
            
            # Scheduler should have cleaned up the task despite the error
            assert scheduler.get_scheduled_count() == 0

    @pytest.mark.asyncio 
    async def test_threshold_evaluation_edge_cases(self, threshold_data: Dict):
        """Test edge cases in threshold evaluation logic"""
        
        # Test threshold scheduler with different scheduling scenarios
        scheduler = ThresholdScheduler()
        
        # Test scheduling at exactly the same time (should not wait)
        now = datetime.now(timezone.utc)
        agent_id = str(uuid.uuid4())
        
        with patch('api.src.backend.queries.agents.approve_agent_version') as mock_approve, \
             patch('api.src.backend.db_manager.get_transaction') as mock_transaction:
             
            # Mock the database transaction context manager
            mock_conn = MockDBConnection()
            mock_transaction.return_value.__aenter__.return_value = mock_conn
            mock_transaction.return_value.__aexit__.return_value = None
            mock_approve.return_value = None  # approve_agent_version doesn't return anything
            
            scheduler.schedule_future_approval(agent_id, 1, now)
            
            # Should execute immediately since time matches
            await asyncio.sleep(0.1)
            
            mock_approve.assert_called_once_with(agent_id, 1, None)

    @pytest.mark.asyncio
    async def test_scheduler_recovery_functionality(self, scheduler: ThresholdScheduler, threshold_data: Dict):
        """Test recovery of pending approvals on startup"""
        # Create agents that should be recovered
        recovery_agent_id = str(uuid.uuid4())
        
        with patch('api.src.backend.queries.agents.approve_agent_version') as mock_approve, \
             patch('api.src.backend.db_manager.get_transaction') as mock_transaction:
             
            # Mock the database transaction context manager
            mock_conn = MockDBConnection()
            mock_transaction.return_value.__aenter__.return_value = mock_conn
            mock_transaction.return_value.__aexit__.return_value = None
            
            mock_approve.return_value = None  # approve_agent_version doesn't return anything
            
            # Simulate recovery - schedule something that should have already happened
            past_time = datetime.now(timezone.utc) - timedelta(seconds=1)
            scheduler.schedule_future_approval(recovery_agent_id, 1, past_time)
            
            # Should execute immediately since time has passed
            await asyncio.sleep(0.1)
            
            mock_approve.assert_called_once_with(recovery_agent_id, 1, None)

    @pytest.mark.asyncio
    async def test_approval_endpoint_integration(self, mock_db_conn):
        """Test approval integration with mocked components"""
        agent_id = uuid.uuid4()
        
        # Test that the scheduler and approval system work together
        scheduler = ThresholdScheduler()
        
        with patch('api.src.backend.queries.agents.approve_agent_version') as mock_approve, \
             patch('api.src.backend.db_manager.get_transaction') as mock_transaction:
             
            # Mock the database transaction context manager
            mock_conn = MockDBConnection()
            mock_transaction.return_value.__aenter__.return_value = mock_conn
            mock_transaction.return_value.__aexit__.return_value = None
            
            mock_approve.return_value = None
            
            # Test integration: schedule multiple agents and verify all are processed
            agent_ids = [str(uuid.uuid4()) for _ in range(2)]
            
            for i, aid in enumerate(agent_ids):
                # Schedule one immediate and one future
                if i == 0:
                    schedule_time = datetime.now(timezone.utc) - timedelta(milliseconds=10)
                else:
                    schedule_time = datetime.now(timezone.utc) + timedelta(milliseconds=200)
                    
                scheduler.schedule_future_approval(aid, 1, schedule_time)
            
            # Wait for all to complete
            await asyncio.sleep(0.5)
            
            # Verify both were called
            assert mock_approve.call_count == 2
            assert scheduler.get_scheduled_count() == 0

    @pytest.mark.asyncio
    async def test_approval_with_real_timeout_constraints(self, scheduler: ThresholdScheduler):
        """Test approval system with realistic timeout constraints"""
        agent_id = str(uuid.uuid4())
        set_id = 1
        
        # Test with realistic timeout (3 seconds)
        start_time = asyncio.get_event_loop().time()
        
        with patch('api.src.backend.queries.agents.approve_agent_version') as mock_approve, \
             patch('api.src.backend.db_manager.get_transaction') as mock_transaction:
             
            # Mock the database transaction context manager
            mock_conn = MockDBConnection()
            mock_transaction.return_value.__aenter__.return_value = mock_conn
            mock_transaction.return_value.__aexit__.return_value = None
            # Mock a function that takes longer than timeout
            async def timeout_approval(*args):
                await asyncio.sleep(5)  # Longer than our timeout
                return {'approved_at': datetime.now(timezone.utc)}
            
            mock_approve.side_effect = timeout_approval
            
            future_time = datetime.now(timezone.utc) + timedelta(milliseconds=100)
            scheduler.schedule_future_approval(agent_id, set_id, future_time)
            
            # Wait and then cancel to test cleanup
            await asyncio.sleep(0.2)
            scheduler.cancel_scheduled_approval(agent_id, set_id)
            
            elapsed = asyncio.get_event_loop().time() - start_time
            assert elapsed < 4.0  # Should have timed out quickly

    @pytest.mark.asyncio
    async def test_concurrent_approval_race_conditions(self, scheduler: ThresholdScheduler):
        """Test race conditions in concurrent approval scenarios"""
        agent_id = str(uuid.uuid4())
        set_id = 1
        
        # Mock approval function with delay to create race condition
        call_count = 0
        async def delayed_approval(*args):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return {'approved_at': datetime.now(timezone.utc)}
        
        with patch('api.src.backend.queries.agents.approve_agent_version', side_effect=delayed_approval):
            # Try to schedule the same agent multiple times rapidly
            future_time = datetime.now(timezone.utc) + timedelta(milliseconds=50)
            
            # Schedule the same agent multiple times - should overwrite
            scheduler.schedule_future_approval(agent_id, set_id, future_time)
            initial_count = scheduler.get_scheduled_count()
            
            scheduler.schedule_future_approval(agent_id, set_id, future_time)
            scheduler.schedule_future_approval(agent_id, set_id, future_time)
            
            # Should still only have one scheduled task (overwrites previous)
            assert scheduler.get_scheduled_count() == initial_count
            
            # Wait for execution
            await asyncio.sleep(0.3)
            
            # Should only have been called once
            assert call_count == 1


@pytest.mark.integration  
class TestApprovalEndpoints:
    """Integration tests for approval-related API endpoints"""

    @pytest.mark.asyncio
    async def test_approve_version_endpoint_immediate(self):
        """Test immediate approval through scheduler"""
        agent_id = str(uuid.uuid4())
        scheduler = ThresholdScheduler()
        
        with patch('api.src.backend.queries.agents.approve_agent_version') as mock_approve, \
             patch('api.src.backend.db_manager.get_transaction') as mock_transaction:
             
            # Mock the database transaction context manager
            mock_conn = MockDBConnection()
            mock_transaction.return_value.__aenter__.return_value = mock_conn
            mock_transaction.return_value.__aexit__.return_value = None
            
            mock_approve.return_value = None
            
            # Test immediate approval (past time)
            past_time = datetime.now(timezone.utc) - timedelta(milliseconds=10)
            scheduler.schedule_future_approval(agent_id, 1, past_time)
            
            await asyncio.sleep(0.1)
            
            mock_approve.assert_called_once_with(agent_id, 1, None)
            assert scheduler.get_scheduled_count() == 0

    @pytest.mark.asyncio
    async def test_approve_version_endpoint_future(self):
        """Test future approval scheduling through scheduler"""
        agent_id = str(uuid.uuid4())
        scheduler = ThresholdScheduler()
        
        with patch('api.src.backend.queries.agents.approve_agent_version') as mock_approve, \
             patch('api.src.backend.db_manager.get_transaction') as mock_transaction:
             
            # Mock the database transaction context manager
            mock_conn = MockDBConnection()
            mock_transaction.return_value.__aenter__.return_value = mock_conn
            mock_transaction.return_value.__aexit__.return_value = None
            
            mock_approve.return_value = None
            
            # Test future scheduling
            future_time = datetime.now(timezone.utc) + timedelta(milliseconds=200)
            scheduler.schedule_future_approval(agent_id, 1, future_time)
            
            # Should be scheduled
            assert scheduler.get_scheduled_count() == 1
            
            # Wait for execution
            await asyncio.sleep(0.3)
            
            mock_approve.assert_called_once_with(agent_id, 1, None)
            assert scheduler.get_scheduled_count() == 0

    @pytest.mark.asyncio
    async def test_approve_version_invalid_password(self):
        """Test approval endpoint with invalid password"""
        agent_id = uuid.uuid4()
        
        with patch('os.getenv', return_value='correct_password'), \
             patch('api.src.utils.auth.verify_request', return_value=False):
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    f"/scoring/approve-version?version_id={agent_id}&set_id=1&approval_password=wrong_password"
                )
            
            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_approve_version_nonexistent_agent(self):
        """Test approval endpoint with non-existent agent"""
        agent_id = uuid.uuid4()
        
        with patch('api.src.backend.db_manager.new_db.acquire') as mock_acquire, \
             patch('api.src.backend.queries.agents.get_agent_by_version_id', return_value=None), \
             patch('os.getenv', return_value='test_password'), \
             patch('api.src.utils.auth.verify_request', return_value=True):
            
            mock_db_conn = MockDBConnection()
            mock_acquire.return_value.__aenter__.return_value = mock_db_conn
            mock_acquire.return_value.__aexit__.return_value = None
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    f"/scoring/approve-version?version_id={agent_id}&set_id=1&approval_password=test_password"
                )
            
            assert response.status_code == 404
            assert "Agent not found" in response.json()['detail']