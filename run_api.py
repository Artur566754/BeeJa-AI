#!/usr/bin/env python3
"""
Startup script for the Server Management API.

This script starts the FastAPI server with uvicorn, handling command-line
arguments for configuration.
"""

import argparse
import os
import sys
import uvicorn

from api.config import APIConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Start the Server Management API server"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=APIConfig.HOST,
        help=f"Host to bind to (default: {APIConfig.HOST})"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=APIConfig.PORT,
        help=f"Port to bind to (default: {APIConfig.PORT})"
    )
    
    parser.add_argument(
        "--api-keys",
        type=str,
        help="Comma-separated list of API keys (overrides API_KEYS environment variable)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level (default: info)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the API server."""
    args = parse_args()
    
    # Set API keys if provided
    if args.api_keys:
        os.environ["API_KEYS"] = args.api_keys
        APIConfig.API_KEYS = args.api_keys.split(",")
    
    # Validate configuration
    try:
        APIConfig.validate_config()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        print("\nPlease set API_KEYS environment variable or use --api-keys argument", file=sys.stderr)
        print("Example: python run_api.py --api-keys your_secret_key_here", file=sys.stderr)
        sys.exit(1)
    
    # Print startup information
    print("=" * 60)
    print("Server Management API")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"API Keys: {len(APIConfig.API_KEYS)} configured")
    print(f"Max Concurrent Sessions: {APIConfig.MAX_CONCURRENT_SESSIONS}")
    print(f"Models Directory: {APIConfig.MODELS_DIR}")
    print(f"Datasets Directory: {APIConfig.DATASETS_DIR}")
    print(f"Database: {APIConfig.DATABASE_PATH}")
    print("=" * 60)
    print(f"\nAPI Documentation: http://{args.host}:{args.port}/docs")
    print(f"Health Check: http://{args.host}:{args.port}/health")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60)
    
    # Start the server
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
