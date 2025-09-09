# Multi-Robot Coordination Framework Docker Image
FROM ros:humble-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV ROBOT_ID=""
ENV MASTER_IP="localhost"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libyaml-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Create workspace
WORKDIR /opt/multi_robot_ws

# Copy source code
COPY . /opt/multi_robot_ws/src/multi_robot_coordination/

# Install the package
RUN cd /opt/multi_robot_ws/src/multi_robot_coordination && \
    pip3 install -e .

# Build ROS workspace
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && \
    cd /opt/multi_robot_ws && \
    colcon build --packages-select multi_robot_coordination"

# Create entrypoint script
RUN echo '#!/bin/bash\n\
source /opt/ros/$ROS_DISTRO/setup.bash\n\
source /opt/multi_robot_ws/install/setup.bash\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set up logging directory
RUN mkdir -p /opt/multi_robot_ws/logs && \
    chmod 777 /opt/multi_robot_ws/logs

# Expose ports
EXPOSE 11311 8080 8081

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python3", "/opt/multi_robot_ws/src/multi_robot_coordination/coordination_master.py", "--robots", "5"]
