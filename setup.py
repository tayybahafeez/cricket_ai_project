#!/usr/bin/env python3
"""
Setup script for Cricket AI Project
"""
import os
import subprocess
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    dirs = [
        "models",
        "dataset",
        "dataset/visuals"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Cricket AI Environment Variables
# Google Gemini API Key (optional - for LLM explanations)
# Get your free API key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=

# Environment settings
ENV=development
LOG_LEVEL=INFO
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")
        print("📝 Please edit .env file and add your GOOGLE_API_KEY if you want LLM explanations")
    else:
        print("✅ .env file already exists")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False
    return True

def main():
    print("🏏 Cricket AI Project Setup")
    print("=" * 40)
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file and add your GOOGLE_API_KEY (optional)")
    print("2. Run: python run_pipeline.py")
    print("3. Visit: http://127.0.0.1:8000/docs")

if __name__ == "__main__":
    main()
