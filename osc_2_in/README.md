# WaLTER Sr's operational-space-control
Operational Space Controller using MuJoCo

## 1. Project Setup and Installation Guide
This guide provides step-by-step instructions for setting up the development environment, installing dependencies, and building the project.

### 1. System Prerequisites

First, ensure your system is up-to-date and has the necessary dependencies.Update your system's package list and upgrade existing packages.
```
sudo apt update
sudo apt upgrade
```
### 2. Check your system architecture
This is important for downloading the correct Bazelisk package.
```
dpkg --print-architecture
lscpu
```
### 3. Install development libraries for graphics and Python 
These are required for the Mujoco viewer and various Python modules.
```
sudo apt install libgl1-mesa-dev xorg-dev
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12-dev python3.12-venv
```
### 4. Install Bazelisk
Bazelisk is a wrapper for Bazel that automatically downloads the correct version.
```
# Navigate to your Downloads directory
cd Downloads/
# Download the appropriate Bazelisk package (e.g., for amd64)
# The user should replace this with their correct architecture if different
# Example command:
# wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64
```
### Install the downloaded .deb package
```
sudo apt install ./bazelisk-amd64.deb
```
### Verify the installation

```
bazel --version
```
Note: If you encounter permissions errors during the Bazelisk installation, you may need to run the following commands to fix the permissions of your apt cache. This is a rare occurrence but can be helpful for troubleshooting.
```
sudo chown -Rv _apt:root /var/cache/apt/archives/partial/
sudo chmod -Rv 700 /var/cache/apt/archives/partial/
sudo apt install ./bazelisk-amd64.deb
```

## 2. Project Setup and Virtual Environment

This section covers cloning the necessary repositories and setting up a dedicated Python virtual environment.


### 1. Clone the project repositories. 

Navigate to the directory where you want to store the code.# Navigate to your desired working directory (e.g., your home folder)
```
cd ..
ls -al
git clone git@github.com:vannem95/mujoco-models.git
git clone git@github.com:vannem95/operational-space-control.git
```

### 2. Create and activate a Python virtual environment.
```
python3.12 -m venv mujoco_env
source mujoco_env/bin/activate
```
Note: Your command prompt should now show (mujoco_env) to indicate the environment is active. You must run this command in any new terminal session to work on the project.


## 3. Install Python Dependencies

With the virtual environment active, install all the required Python libraries.

1. Upgrade pip and install core dependencies.
```
pip install --upgrade pip
pip install numpy==2.2.2 mujoco==3.2.7
pip install absl-py==2.1.0
pip install PyYAML==6.0.2
pip install casadi==3.6.7
pip install etils==1.11.0
pip install fsspec==2025.2.0
pip install glfw==2.8.0
pip install importlib_resources==6.5.2d
pip install PyOpenGL==3.1.9
pip install typing_extensions==4.12.2
pip install zipp==3.21.0
```
## 4. Build and Run the Project

Follow these steps to generate project files and run the applications.

### 1. Navigate to the operational-space-control directory.
```
cd operational-space-control/
```

### 2. Run the autogen scripts. 

These scripts generate necessary files for the controllers and models.
```
python3.12 walter_sr_autogen.py
cd go2_files/
python3.12 go2_autogen.py
cd ..
```

### 3. Build the project with Bazel.
```
bazel build //operational-space-control/walter_sr/autogen:autogen_defines_cc
```

### 4. Run the autogen rule and a standing example
```
bazel run //operational-space-control/walter_sr/autogen:autogen_defines_cc
bazel run //examples:walter_sr_standing
```