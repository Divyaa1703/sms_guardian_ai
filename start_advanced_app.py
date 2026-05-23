#!/usr/bin/env python3
"""
Start the advanced SMS Spam Classifier with all features:
- Home page
- User registration
- User login
- Dashboard
- SMS Classification
- History tracking
"""

import subprocess
import sys
import webbrowser
import time
import os

def start_app():
    """Start the advanced Streamlit app"""
    print("🚀 Starting Advanced SMS Spam Classifier")
    print("=" * 50)
    print("Features included:")
    print("✓ Beautiful Home Page")
    print("✓ User Registration")
    print("✓ User Login")
    print("✓ Personal Dashboard")
    print("✓ SMS Classification")
    print("✓ Classification History")
    print("✓ Analytics & Charts")
    print("=" * 50)
    
    try:
        # Start the Streamlit app
        print("Starting server...")
        cmd = [sys.executable, "-m", "streamlit", "run", "main_app.py", 
               "--server.port", "8501", 
               "--server.headless", "false",
               "--browser.gatherUsageStats", "false"]
        
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        # Wait a moment for the server to start
        print("Waiting for server to start...")
        time.sleep(3)
        
        # Open browser
        print("Opening browser...")
        webbrowser.open("http://localhost:8501")
        
        print("\n" + "=" * 50)
        print("🎉 SMS Spam Classifier is now running!")
        print("📱 URL: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Wait for the process
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            process.terminate()
            print("✅ Server stopped successfully!")
            
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_app()
