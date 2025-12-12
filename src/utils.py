"""
Utilities module for XENO Bot
Handles logging and timing functionality
"""
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict

# ===== Configure Logging =====
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_exception(exc_type, exc_value, exc_traceback):
    """Log uncaught exceptions"""
    if issubclass(exc_type, KeyboardInterrupt):
        return
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = log_exception
logging.info("App started successfully.")


# ===== Time Tracking Class =====
class PipelineTimer:
    """Timer for tracking pipeline execution steps"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all timing data for a new request"""
        self.start_time = time.time()
        self.step_times = {}
        self.step_start = None
        self.current_step = None
    
    @contextmanager
    def time_step(self, step_name: str):
        """Context manager to time a specific step"""
        step_start = time.time()
        self.current_step = step_name
        try:
            yield
        finally:
            step_end = time.time()
            self.step_times[step_name] = round((step_end - step_start) * 1000, 2)  # Convert to milliseconds
            self.current_step = None
    
    def get_total_time(self):
        """Get total elapsed time since reset"""
        return round((time.time() - self.start_time) * 1000, 2)
    
    def get_timing_summary(self) -> Dict:
        """Get a summary of all timing data"""
        total_time = self.get_total_time()
        return {
            'total_time_ms': total_time,
            'step_times': self.step_times,
            'timestamp': datetime.now().isoformat()
        }
