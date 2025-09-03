#!/usr/bin/env python3
"""
Cricket AI Pipeline Runner
Complete pipeline to run EDA, training, and start API server
"""
import sys
from pathlib import Path
import subprocess
import argparse

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Cricket AI Pipeline Runner")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA step")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    parser.add_argument("--skip-server", action="store_true", help="Skip starting server")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    
    args = parser.parse_args()
    
    print("🏏 Cricket AI Pipeline Starting...")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = Path("cricket_ai/datasets/cricket_dataset.csv")
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please ensure the dataset is in the correct location")
        sys.exit(1)
    
    success = True
    
    # Step 1: Run EDA
    if not args.skip_eda:
        success = run_command(
            "python -m cricket_ai.pipelines.eda",
            "Exploratory Data Analysis"
        )
        if not success:
            sys.exit(1)
    
    # Step 2: Train Model
    if not args.skip_train:
        success = run_command(
            "python -m cricket_ai.pipelines.train",
            "Model Training"
        )
        if not success:
            sys.exit(1)
    
    # Step 3: Start API Server
    if not args.skip_server:
        print(f"\n🚀 Starting API server on port {args.port}...")
        print("=" * 50)
        print("📖 API Documentation: http://127.0.0.1:8000/docs")
        print("🔍 Alternative Docs: http://127.0.0.1:8000/redoc")
        print("❤️  Health Check: http://127.0.0.1:8000/health")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)
        
        try:
            subprocess.run([
                "uvicorn", "cricket_ai.api.main:app", 
                "--reload", 
                "--host", "127.0.0.1", 
                "--port", str(args.port)
            ])
        except KeyboardInterrupt:
            print("\n👋 Server stopped by user")
    
    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()
