"""
System Monitor service for tracking server resource usage.

This module provides the SystemMonitor class that tracks CPU, memory, disk,
and GPU usage statistics. It refreshes metrics in the background every 5 seconds
and provides cached values for fast API responses.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import psutil

from api.models.data_models import SystemInfo
from api.config import APIConfig


logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    Monitors system resources and provides current usage statistics.
    
    The monitor runs a background task that refreshes metrics every 5 seconds.
    GPU metrics are collected using pynvml if available, with graceful fallback
    if GPU is not available or pynvml is not installed.
    """
    
    def __init__(self):
        """Initialize the system monitor."""
        self._cached_info: Optional[SystemInfo] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._gpu_available = False
        self._pynvml_handle = None
        
        # Try to initialize GPU monitoring
        self._init_gpu_monitoring()
    
    def _init_gpu_monitoring(self) -> None:
        """
        Initialize GPU monitoring with pynvml.
        
        Gracefully handles cases where pynvml is not installed or
        no GPU is available.
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            # Try to get the first GPU device
            self._pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._gpu_available = True
            logger.info("GPU monitoring initialized successfully")
        except ImportError:
            logger.info("pynvml not installed, GPU monitoring disabled")
            self._gpu_available = False
        except Exception as e:
            logger.info(f"GPU not available or error initializing: {e}")
            self._gpu_available = False
    
    def _get_gpu_metrics(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Get current GPU usage metrics.
        
        Returns:
            Tuple of (gpu_usage_percent, gpu_memory_used_mb, gpu_memory_total_mb)
            Returns (None, None, None) if GPU is not available.
        """
        if not self._gpu_available:
            return None, None, None
        
        try:
            import pynvml
            
            # Get GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self._pynvml_handle)
            gpu_usage_percent = float(utilization.gpu)
            
            # Get GPU memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._pynvml_handle)
            gpu_memory_used_mb = mem_info.used / (1024 * 1024)  # Convert bytes to MB
            gpu_memory_total_mb = mem_info.total / (1024 * 1024)  # Convert bytes to MB
            
            return gpu_usage_percent, gpu_memory_used_mb, gpu_memory_total_mb
        except Exception as e:
            logger.warning(f"Error getting GPU metrics: {e}")
            return None, None, None
    
    def _collect_system_info(self) -> SystemInfo:
        """
        Collect current system resource information.
        
        Returns:
            SystemInfo object with current metrics.
        """
        # CPU usage
        cpu_usage_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024 * 1024)  # Convert bytes to MB
        memory_total_mb = memory.total / (1024 * 1024)  # Convert bytes to MB
        memory_percent = memory.percent
        
        # Disk usage (for the root partition)
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024 * 1024 * 1024)  # Convert bytes to GB
        disk_total_gb = disk.total / (1024 * 1024 * 1024)  # Convert bytes to GB
        
        # GPU metrics
        gpu_usage_percent, gpu_memory_used_mb, gpu_memory_total_mb = self._get_gpu_metrics()
        
        return SystemInfo(
            cpu_usage_percent=cpu_usage_percent,
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            memory_percent=memory_percent,
            disk_free_gb=disk_free_gb,
            disk_total_gb=disk_total_gb,
            gpu_available=self._gpu_available,
            gpu_usage_percent=gpu_usage_percent,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def _monitoring_loop(self) -> None:
        """
        Background task that refreshes system metrics every 5 seconds.
        
        This loop runs continuously until stop_monitoring() is called.
        """
        logger.info("System monitoring loop started")
        while True:
            try:
                self._cached_info = self._collect_system_info()
                await asyncio.sleep(APIConfig.MONITOR_REFRESH_INTERVAL)
            except asyncio.CancelledError:
                logger.info("System monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(APIConfig.MONITOR_REFRESH_INTERVAL)
    
    def start_monitoring(self) -> None:
        """
        Start the background monitoring task.
        
        This should be called when the API server starts up.
        """
        if self._monitoring_task is not None and not self._monitoring_task.done():
            logger.warning("Monitoring task already running")
            return
        
        # Collect initial metrics immediately
        self._cached_info = self._collect_system_info()
        
        # Start background task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """
        Stop the background monitoring task.
        
        This should be called when the API server shuts down.
        """
        if self._monitoring_task is not None and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            logger.info("System monitoring stopped")
    
    def get_system_info(self) -> SystemInfo:
        """
        Get the current system resource information.
        
        Returns cached metrics from the background monitoring task.
        If monitoring hasn't started yet, collects metrics immediately.
        
        Returns:
            SystemInfo object with current metrics.
        """
        if self._cached_info is None:
            # If monitoring hasn't started or no cached data, collect immediately
            self._cached_info = self._collect_system_info()
        
        return self._cached_info
    
    def __del__(self):
        """Cleanup GPU monitoring on deletion."""
        if self._gpu_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass


# Global instance for use across the API
_system_monitor: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """
    Get the global SystemMonitor instance.
    
    Creates the instance on first call (singleton pattern).
    
    Returns:
        The global SystemMonitor instance.
    """
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor
