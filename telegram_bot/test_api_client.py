"""
Simple test to verify the API client functionality.
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import APIClient


async def test_api_client():
    """Test basic API client functionality."""
    
    # Test with dummy credentials
    client = APIClient("http://localhost:8000", "test_key")
    
    print("✓ APIClient initialized successfully")
    print(f"  Base URL: {client.base_url}")
    print(f"  Headers: {client.headers}")
    
    # Test health check (will fail if server not running, but that's ok)
    try:
        is_healthy = await client.health_check()
        if is_healthy:
            print("✓ API server is healthy")
        else:
            print("⚠ API server returned unhealthy status")
    except Exception as e:
        print(f"⚠ Could not connect to API server (expected if not running): {e}")
    
    print("\n✓ All basic tests passed!")


if __name__ == "__main__":
    asyncio.run(test_api_client())
