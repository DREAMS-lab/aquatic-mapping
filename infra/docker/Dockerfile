# Aquatic Mapping Simulation Container
# ROS2 Jazzy + Gazebo Harmonic + PX4 SITL + Micro XRCE DDS Agent
#
# Build: docker build -t aquatic-sim .
# Run:   docker run -d --name sim1 -e TRIAL_ID=1 aquatic-sim

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ============================================================================
# Base system and dependencies
# ============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget curl gnupg2 lsb-release software-properties-common \
    python3 python3-pip python3-venv python3-dev \
    net-tools iputils-ping dnsutils vim tmux htop \
    xvfb x11vnc novnc websockify fluxbox xterm \
    locales tzdata sudo ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ============================================================================
# ROS 2 Jazzy
# ============================================================================

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-desktop ros-jazzy-ros-base ros-jazzy-geometry-msgs ros-jazzy-sensor-msgs \
    ros-jazzy-nav-msgs ros-jazzy-std-msgs ros-jazzy-std-srvs ros-jazzy-tf2 ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs ros-jazzy-robot-state-publisher ros-jazzy-joint-state-publisher \
    ros-jazzy-rviz2 ros-jazzy-rosbag2 ros-jazzy-rosbag2-storage-mcap ros-dev-tools \
    python3-colcon-common-extensions python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true
RUN rosdep update --rosdistro jazzy

# ============================================================================
# Gazebo Harmonic
# ============================================================================

RUN curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

RUN apt-get update && apt-get install -y --no-install-recommends \
    gz-harmonic ros-jazzy-ros-gz \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# PX4 Dependencies
# ============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    astyle ninja-build exiftool genromfs libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav libeigen3-dev libopencv-dev \
    libxml2-dev libxslt1-dev unzip zip default-jdk ant \
    && rm -rf /var/lib/apt/lists/*

# PX4 build dependencies (these are needed for PX4 build, keep system-wide)
RUN pip3 install --break-system-packages \
    kconfiglib jinja2 jsonschema pyros-genmsg packaging toml future empy==3.3.4 pyyaml

# Create virtual environments for planners
RUN python3 -m venv /opt/venv/exact && \
    /opt/venv/exact/bin/pip install --upgrade pip && \
    /opt/venv/exact/bin/pip install numpy scipy matplotlib scikit-learn && \
    /opt/venv/exact/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    /opt/venv/exact/bin/pip install gpytorch

RUN python3 -m venv /opt/venv/pose_aware && \
    /opt/venv/pose_aware/bin/pip install --upgrade pip && \
    /opt/venv/pose_aware/bin/pip install numpy scipy matplotlib scikit-learn && \
    /opt/venv/pose_aware/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    /opt/venv/pose_aware/bin/pip install gpytorch

# ============================================================================
# Micro XRCE DDS Agent (from source)
# ============================================================================

USER root
WORKDIR /tmp

# Build Micro-XRCE-DDS-Agent from source (v3.0.1 for PX4 main branch compatibility)
RUN git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git && \
    cd Micro-XRCE-DDS-Agent && \
    git checkout v3.0.1 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /tmp && rm -rf Micro-XRCE-DDS-Agent

# ============================================================================
# Create simulation user
# ============================================================================

RUN useradd -m -s /bin/bash -G sudo simuser
RUN echo "simuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER simuser
WORKDIR /home/simuser

# ============================================================================
# PX4 SITL
# ============================================================================

RUN git clone --recursive https://github.com/PX4/PX4-Autopilot.git /home/simuser/PX4-Autopilot

WORKDIR /home/simuser/PX4-Autopilot
RUN git submodule update --init --recursive
RUN make px4_sitl

# ============================================================================
# Aquatic Mapping
# ============================================================================

RUN git clone --recursive https://github.com/DREAMS-lab/aquatic-mapping.git /home/simuser/aquatic-mapping

WORKDIR /home/simuser/aquatic-mapping

RUN mkdir -p src/sampling/data/missions

RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
    colcon build && \
    source install/setup.bash"

# ============================================================================
# VNC Setup
# ============================================================================

USER root

RUN mkdir -p /home/simuser/.vnc && \
    x11vnc -storepasswd "" /home/simuser/.vnc/passwd && \
    chown -R simuser:simuser /home/simuser/.vnc

RUN ln -s /usr/share/novnc/vnc.html /usr/share/novnc/index.html || true

# ============================================================================
# Entrypoint and startup scripts
# ============================================================================

COPY entrypoint-simple.sh /entrypoint.sh
COPY start-mission-headless.sh /home/simuser/start-mission-headless.sh
COPY mission-monitor.sh /home/simuser/mission-monitor.sh
RUN chmod 755 /entrypoint.sh && chown simuser:simuser /entrypoint.sh && \
    chmod 755 /home/simuser/start-mission-headless.sh && chown simuser:simuser /home/simuser/start-mission-headless.sh && \
    chmod 755 /home/simuser/mission-monitor.sh && chown simuser:simuser /home/simuser/mission-monitor.sh

# Environment variables
ENV DISPLAY=:99
ENV ROS_DOMAIN_ID=0
ENV TRIAL_ID=1
ENV HEADLESS=1
ENV VNC_PORT=5900
ENV NOVNC_PORT=6080
ENV PX4_HOME=/home/simuser/PX4-Autopilot
ENV AQUATIC_WORKSPACE=/home/simuser/aquatic-mapping

EXPOSE 5900 6080 14540 8888

USER simuser
WORKDIR /home/simuser

ENTRYPOINT ["/entrypoint.sh"]
CMD ["mission"]
