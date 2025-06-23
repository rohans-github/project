#!/usr/bin/env python3
"""
Quick start script for AI Personal Finance Assistant
Run this script to launch the application with proper setup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'logs', 'backups']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def launch_application():
    """Launch the Streamlit application"""
    print("\n🚀 Launching AI Personal Finance Assistant...")
    print("🌐 Opening browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
    except Exception as e:
        print(f"❌ Error launching application: {e}")

def main():
    """Main function to set up and run the application"""
    print("🏦 AI Personal Finance Assistant - Setup & Launch")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check if main.py exists
    if not Path("main.py").exists():
        print("❌ main.py not found. Please run this script from the project root directory.")
        return
    
    # Check dependencies
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"\n📋 Missing packages: {', '.join(missing_packages)}")
        
        # Ask user if they want to install dependencies
        response = input("\n🤔 Would you like to install missing dependencies? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            if not install_dependencies():
                print("❌ Failed to install dependencies. Please install manually.")
                return
        else:
            print("❌ Cannot run without required dependencies.")
            print("💡 Install them manually with: pip install -r requirements.txt")
            return
    
    # Create necessary directories
    print("\n📁 Setting up directories...")
    create_directories()
    
    # Launch application
    launch_application()

def show_help():
    """Show help information"""
    help_text = """
    🏦 AI Personal Finance Assistant - Quick Start Guide
    
    This script helps you set up and launch the personal finance assistant.
    
    Usage:
        python run.py          # Launch the application
        python run.py --help   # Show this help message
    
    What this script does:
    1. ✅ Checks Python version compatibility (3.8+)
    2. 📦 Verifies required packages are installed
    3. 🔧 Installs missing dependencies if needed
    4. 📁 Creates necessary directories
    5. 🚀 Launches the Streamlit web application
    
    Manual Setup (Alternative):
    1. pip install -r requirements.txt
    2. streamlit run main.py
    
    Troubleshooting:
    - Ensure you're in the project root directory
    - Check that Python 3.8+ is installed
    - Try running: pip install --upgrade pip
    - For M1 Macs, you might need: pip install --upgrade pip setuptools wheel
    
    """
    print(help_text)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
    else:
        main()
