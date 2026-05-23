#!/usr/bin/env python3
"""
Launcher script for the SMS Spam Classifier Streamlit app
This script handles the streamlit command not being in PATH
"""

import subprocess
import sys
import os

def run_streamlit_app(app_file="app.py"):
    """Run the streamlit app using python -m streamlit"""
    try:
        print(f"Starting SMS Spam Classifier ({app_file})...")
        print("=" * 50)
        
        # Check if the app file exists
        if not os.path.exists(app_file):
            print(f"Error: {app_file} not found in current directory")
            return False
        
        # Run streamlit using python -m
        cmd = [sys.executable, "-m", "streamlit", "run", app_file]
        print(f"Running command: {' '.join(cmd)}")
        print("=" * 50)
        
        # Start the streamlit server
        process = subprocess.run(cmd)
        return True
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        return True
    except Exception as e:
        print(f"Error running streamlit: {e}")
        return False

def main():
    """Main function"""
    print("SMS Spam Classifier Launcher")
    print("=" * 50)
    
    # Check available apps
    apps = []
    if os.path.exists("app.py"):
        apps.append(("app.py", "Simple SMS Classifier"))
    if os.path.exists("main_app.py"):
        apps.append(("main_app.py", "Advanced SMS Classifier with Dashboard"))
    
    if not apps:
        print("No Streamlit apps found!")
        return
    
    if len(apps) == 1:
        # Only one app available
        app_file, description = apps[0]
        print(f"Found: {description}")
        run_streamlit_app(app_file)
    else:
        # Prefer the advanced app when both are present
        apps.sort(key=lambda item: 0 if item[0] == 'main_app.py' else 1)
        print("Available apps:")
        for i, (app_file, description) in enumerate(apps, 1):
            print(f"{i}. {description} ({app_file})")
        
        try:
            default_choice_label = next((i for i, (app_file, _) in enumerate(apps, 1) if app_file == 'main_app.py'), 1)
            choice = input(f"\nChoose an app (1-{len(apps)}) or press Enter for advanced app ({default_choice_label}): ").strip()
            
            if choice == "":
                choice_idx = default_choice_label - 1
            else:
                choice_idx = int(choice) - 1

            if 0 <= choice_idx < len(apps):
                app_file, description = apps[choice_idx]
                print(f"\nStarting: {description}")
                run_streamlit_app(app_file)
            else:
                print("Invalid choice!")
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")

if __name__ == "__main__":
    main()
