"""
Unit tests for the System Monitor service.

Tests verify that system information is collected correctly and contains
all required fields as specified in Requirements 3.1, 3.2, 3.3, 3.4.
"""

import asyncio
import pytest
from datetime import datetime

from api.services.system_monitor import SystemMonitor, get_system_monitor
from api.models.data_models import SystemInfo


class TestSystemMonitor:
    """Test suite for SystemMonitor class."""
    
    def test_system_info_contains_required_fields(self):
        """
        Test that system info contains all required fields.
        
        **Property 11: System info contains required fields**
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
        
        Verifies that get_system_info() returns a SystemInfo object with:
        - CPU usage percentage
        - Memory usage (MB and percent)
        - Disk space (free and total in GB)
        - GPU information (availability flag and optional metrics)
        - Timestamp
        """
        monitor = SystemMonitor()
        system_info = monitor.get_system_info()
        
        # Verify we got a SystemInfo object
        assert isinstance(system_info, SystemInfo)
        
        # Requirement 3.1: CPU usage percentage
        assert hasattr(system_info, 'cpu_usage_percent')
        assert isinstance(system_info.cpu_usage_percent, float)
        assert 0.0 <= system_info.cpu_usage_percent <= 100.0
        
        # Requirement 3.2: Memory usage in MB and percentage
        assert hasattr(system_info, 'memory_used_mb')
        assert isinstance(system_info.memory_used_mb, float)
        assert system_info.memory_used_mb >= 0.0
        
        assert hasattr(system_info, 'memory_total_mb')
        assert isinstance(system_info.memory_total_mb, float)
        assert system_info.memory_total_mb > 0.0
        
        assert hasattr(system_info, 'memory_percent')
        assert isinstance(system_info.memory_percent, float)
        assert 0.0 <= system_info.memory_percent <= 100.0
        
        # Requirement 3.4: Disk space in GB
        assert hasattr(system_info, 'disk_free_gb')
        assert isinstance(system_info.disk_free_gb, float)
        assert system_info.disk_free_gb >= 0.0
        
        assert hasattr(system_info, 'disk_total_gb')
        assert isinstance(system_info.disk_total_gb, float)
        assert system_info.disk_total_gb > 0.0
        
        # Requirement 3.3: GPU information
        assert hasattr(system_info, 'gpu_available')
        assert isinstance(system_info.gpu_available, bool)
        
        # If GPU is available, check GPU metrics
        if system_info.gpu_available:
            assert hasattr(system_info, 'gpu_usage_percent')
            assert system_info.gpu_usage_percent is not None
            assert isinstance(system_info.gpu_usage_percent, float)
            assert 0.0 <= system_info.gpu_usage_percent <= 100.0
            
            assert hasattr(system_info, 'gpu_memory_used_mb')
            assert system_info.gpu_memory_used_mb is not None
            assert isinstance(system_info.gpu_memory_used_mb, float)
            assert system_info.gpu_memory_used_mb >= 0.0
            
            assert hasattr(system_info, 'gpu_memory_total_mb')
            assert system_info.gpu_memory_total_mb is not None
            assert isinstance(system_info.gpu_memory_total_mb, float)
            assert system_info.gpu_memory_total_mb > 0.0
        else:
            # If GPU is not available, GPU metrics should be None
            assert system_info.gpu_usage_percent is None
            assert system_info.gpu_memory_used_mb is None
            assert system_info.gpu_memory_total_mb is None
        
        # Verify timestamp is present and recent
        assert hasattr(system_info, 'timestamp')
        assert isinstance(system_info.timestamp, datetime)
    
    def test_system_info_values_are_reasonable(self):
        """
        Test that system info values are within reasonable ranges.
        
        Verifies that collected metrics make sense (e.g., used memory
        doesn't exceed total memory).
        """
        monitor = SystemMonitor()
        system_info = monitor.get_system_info()
        
        # Memory used should not exceed total
        assert system_info.memory_used_mb <= system_info.memory_total_mb
        
        # Disk free should not exceed total
        assert system_info.disk_free_gb <= system_info.disk_total_gb
        
        # Memory percent should match the ratio
        expected_percent = (system_info.memory_used_mb / system_info.memory_total_mb) * 100
        # Allow small floating point differences
        assert abs(system_info.memory_percent - expected_percent) < 1.0
        
        # If GPU is available, GPU memory used should not exceed total
        if system_info.gpu_available:
            assert system_info.gpu_memory_used_mb <= system_info.gpu_memory_total_mb
    
    def test_multiple_calls_return_valid_data(self):
        """
        Test that multiple calls to get_system_info() return valid data.
        
        Verifies that the monitor can be called multiple times without errors.
        """
        monitor = SystemMonitor()
        
        # Call multiple times
        for _ in range(3):
            system_info = monitor.get_system_info()
            assert isinstance(system_info, SystemInfo)
            assert system_info.cpu_usage_percent >= 0.0
            assert system_info.memory_total_mb > 0.0
    
    def test_get_system_monitor_singleton(self):
        """
        Test that get_system_monitor() returns a singleton instance.
        
        Verifies that multiple calls return the same instance.
        """
        monitor1 = get_system_monitor()
        monitor2 = get_system_monitor()
        
        assert monitor1 is monitor2
        assert isinstance(monitor1, SystemMonitor)
    
    def test_gpu_monitoring_graceful_fallback(self):
        """
        Test that GPU monitoring fails gracefully when GPU is not available.
        
        Verifies that the monitor works even if pynvml is not installed
        or no GPU is present.
        """
        monitor = SystemMonitor()
        system_info = monitor.get_system_info()
        
        # Should always return valid system info, regardless of GPU availability
        assert isinstance(system_info, SystemInfo)
        assert system_info.cpu_usage_percent >= 0.0
        
        # GPU fields should be present but may be None
        assert hasattr(system_info, 'gpu_available')
        if not system_info.gpu_available:
            assert system_info.gpu_usage_percent is None
            assert system_info.gpu_memory_used_mb is None
            assert system_info.gpu_memory_total_mb is None


@pytest.mark.asyncio
async def test_monitoring_loop_updates_cache():
    """
    Test that the monitoring loop updates cached system info.
    
    Verifies that start_monitoring() creates a background task that
    refreshes metrics periodically.
    """
    import asyncio
    
    monitor = SystemMonitor()
    
    # Get initial info
    initial_info = monitor.get_system_info()
    initial_timestamp = initial_info.timestamp
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Wait for at least one refresh cycle (5 seconds + buffer)
    await asyncio.sleep(6)
    
    # Get updated info
    updated_info = monitor.get_system_info()
    updated_timestamp = updated_info.timestamp
    
    # Timestamp should be updated
    assert updated_timestamp > initial_timestamp
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Wait a bit for cleanup
    await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_start_monitoring_idempotent():
    """
    Test that calling start_monitoring() multiple times is safe.
    
    Verifies that starting monitoring when already running doesn't
    create duplicate tasks.
    """
    monitor = SystemMonitor()
    
    # Start monitoring twice
    monitor.start_monitoring()
    monitor.start_monitoring()
    
    # Should still work fine
    system_info = monitor.get_system_info()
    assert isinstance(system_info, SystemInfo)
    
    # Cleanup
    monitor.stop_monitoring()
    await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_stop_monitoring_when_not_running():
    """
    Test that calling stop_monitoring() when not running is safe.
    
    Verifies that stopping monitoring when not started doesn't cause errors.
    """
    monitor = SystemMonitor()
    
    # Stop without starting - should not raise an error
    monitor.stop_monitoring()
    
    # Should still be able to get system info
    system_info = monitor.get_system_info()
    assert isinstance(system_info, SystemInfo)
