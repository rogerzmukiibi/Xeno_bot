"""
Unit tests for utils module
Tests the PipelineTimer class
"""
import unittest
import time
from src.utils import PipelineTimer


class TestPipelineTimer(unittest.TestCase):
    """Test cases for PipelineTimer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.timer = PipelineTimer()
    
    def test_initialization(self):
        """Test timer initialization"""
        self.assertIsNotNone(self.timer.start_time)
        self.assertEqual(self.timer.step_times, {})
        self.assertIsNone(self.timer.step_start)
        self.assertIsNone(self.timer.current_step)
    
    def test_reset(self):
        """Test timer reset functionality"""
        # Add some data
        self.timer.step_times = {'test': 100}
        self.timer.current_step = 'test'
        
        # Reset
        self.timer.reset()
        
        # Verify reset
        self.assertEqual(self.timer.step_times, {})
        self.assertIsNone(self.timer.current_step)
    
    def test_time_step_context_manager(self):
        """Test timing a step using context manager"""
        with self.timer.time_step('test_step'):
            time.sleep(0.1)  # Sleep for 100ms
        
        # Check that step was timed
        self.assertIn('test_step', self.timer.step_times)
        # Should be approximately 100ms (allowing some variance)
        self.assertGreater(self.timer.step_times['test_step'], 90)
        self.assertLess(self.timer.step_times['test_step'], 150)
    
    def test_multiple_steps(self):
        """Test timing multiple steps"""
        with self.timer.time_step('step1'):
            time.sleep(0.05)
        
        with self.timer.time_step('step2'):
            time.sleep(0.05)
        
        # Both steps should be recorded
        self.assertIn('step1', self.timer.step_times)
        self.assertIn('step2', self.timer.step_times)
        self.assertEqual(len(self.timer.step_times), 2)
    
    def test_get_total_time(self):
        """Test getting total elapsed time"""
        time.sleep(0.1)
        total_time = self.timer.get_total_time()
        
        # Should be at least 100ms
        self.assertGreater(total_time, 90)
    
    def test_get_timing_summary(self):
        """Test getting timing summary"""
        with self.timer.time_step('step1'):
            time.sleep(0.05)
        
        summary = self.timer.get_timing_summary()
        
        # Check summary structure
        self.assertIn('total_time_ms', summary)
        self.assertIn('step_times', summary)
        self.assertIn('timestamp', summary)
        self.assertIn('step1', summary['step_times'])
    
    def test_current_step_tracking(self):
        """Test that current_step is tracked correctly"""
        self.assertIsNone(self.timer.current_step)
        
        with self.timer.time_step('test_step'):
            # During execution, current_step should be set
            self.assertEqual(self.timer.current_step, 'test_step')
        
        # After execution, current_step should be None
        self.assertIsNone(self.timer.current_step)
    
    def test_exception_handling_in_timer(self):
        """Test that timer handles exceptions properly"""
        try:
            with self.timer.time_step('error_step'):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Step should still be recorded even if exception occurred
        self.assertIn('error_step', self.timer.step_times)
        # current_step should be None after context manager exits
        self.assertIsNone(self.timer.current_step)


if __name__ == '__main__':
    unittest.main()
