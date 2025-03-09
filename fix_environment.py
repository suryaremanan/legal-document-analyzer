import subprocess
import sys
import os

def fix_environment():
    """Fix the PyTorch and transformer libraries in the environment"""
    print("Starting environment fix...")
    
    # Uninstall problematic packages
    print("Removing incompatible packages...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", 
                   "torch", "torchvision", "torchaudio", 
                   "transformers", "sentence-transformers"])
    
    # Install compatible versions
    print("Installing compatible versions...")
    subprocess.run([sys.executable, "-m", "pip", "install", "torch==1.11.0"])
    subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.18.0"])
    subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers==2.1.0"])
    
    print("Environment fix completed. Please restart your application.")

if __name__ == "__main__":
    fix_environment() 