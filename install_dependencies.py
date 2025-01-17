import os
import subprocess
import platform

# List of required Python packages
dependencies = [
    "numpy",
    "opencv-python-headless",
    "tensorflow",
    "scikit-learn",
    "streamlit",
    "pillow",
    "matplotlib"
]

def install_dependencies():
    try:
        # Detect the operating system
        os_type = platform.system()
        print(f"Detected OS: {os_type}")

        # Check for Python and pip installation
        print("Checking for Python and pip installation...")
        subprocess.run(["python", "--version"], check=True)
        subprocess.run(["pip", "--version"], check=True)

        # Install dependencies
        print("Installing required Python packages...")
        for package in dependencies:
            subprocess.run(["pip", "install", "--upgrade", package], check=True)

        print("All dependencies installed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}. Please check your Python and pip setup.")

if __name__ == "__main__":
    install_dependencies()
