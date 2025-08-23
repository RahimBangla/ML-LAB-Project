#!/usr/bin/env python3
"""
Startup script for the Heart Disease Prediction System
"""

import os
import sys
import argparse
from app import app, initialize_sample_data

def main():
    """Main function to run the application"""
    
    parser = argparse.ArgumentParser(description='Heart Disease Prediction System')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--init-data', action='store_true', help='Initialize sample dataset on startup')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏥 Heart Disease Prediction System")
    print("=" * 60)
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    print("=" * 60)
    
    try:
        # Initialize sample data if requested
        if args.init_data:
            print("Initializing sample dataset...")
            initialize_sample_data()
            print("✅ Sample dataset initialized successfully!")
        
        # Run the application
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=args.debug
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
