#!/usr/bin/env python3
"""
Setup script for Multi-Robot Coordination Framework
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Get version
def get_version():
    version_file = os.path.join("src", "utils", "version.py")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            exec(f.read())
            return locals()["__version__"]
    return "0.1.0"

setup(
    name="multi-robot-coordination",
    version=get_version(),
    author="Multi-Robot Coordination Team",
    author_email="team@multirobot.ai",
    description="Distributed multi-agent reinforcement learning system for coordinating autonomous robots",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multi-robot-coordination",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "ros2": [
            "rclpy>=3.3.0",
            "std-msgs>=4.2.0",
            "geometry-msgs>=4.2.0",
            "sensor-msgs>=4.2.0",
            "nav-msgs>=4.2.0",
        ],
        "gpu": [
            "cupy>=10.0.0",
            "tensorflow-gpu>=2.8.0",
        ],
        "cloud": [
            "boto3>=1.21.0",
            "google-cloud>=0.34.0",
            "azure-storage-blob>=12.10.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.942",
            "pre-commit>=2.17.0",
        ],
        "all": [
            "rclpy>=3.3.0",
            "std-msgs>=4.2.0",
            "geometry-msgs>=4.2.0",
            "sensor-msgs>=4.2.0",
            "nav-msgs>=4.2.0",
            "cupy>=10.0.0",
            "tensorflow-gpu>=2.8.0",
            "boto3>=1.21.0",
            "google-cloud>=0.34.0",
            "azure-storage-blob>=12.10.0",
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.942",
            "pre-commit>=2.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "multi-robot-master=coordination_master:main",
            "multi-robot-agent=robot_agent:main",
            "multi-robot-monitor=system_monitor:main",
            "multi-robot-tasks=task_generator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.xml", "*.launch", "*.launch.py"],
    },
    zip_safe=False,
    keywords=[
        "robotics",
        "multi-agent",
        "reinforcement-learning",
        "distributed-systems",
        "ros2",
        "coordination",
        "autonomous-robots",
        "task-allocation",
        "fault-tolerance",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/multi-robot-coordination/issues",
        "Source": "https://github.com/yourusername/multi-robot-coordination",
        "Documentation": "https://multi-robot-coordination.readthedocs.io/",
        "Funding": "https://github.com/sponsors/yourusername",
    },
)
