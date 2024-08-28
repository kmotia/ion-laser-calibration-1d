import os
import subprocess
import sys


venv_name = "laser_calibration"

# Define the path for the virtual environment
venv_path = os.path.join(os.getcwd(), venv_name)

# Check if the virtual environment already exists
if not os.path.exists(venv_path):
    print(f"Creating virtual environment '{venv_name}'...")
    subprocess.check_call([sys.executable, "-m", "venv", venv_name])
else:
    print(f"Virtual environment '{venv_name}' already exists.")

# Install the required packages from requirements.txt
subprocess.check_call([os.path.join(venv_path, "bin", "python"), "-m", "pip", "install", "-r", "requirements.txt"])
print("Setup complete. ")