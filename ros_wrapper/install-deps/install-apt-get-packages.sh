ROS_DISTRO=noetic
echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# add-apt-repository -y ppa:ignaciovizzo/opencv3-nonfree

apt-get update && \
    apt-get -y install \
            git \
            libceres-dev \
            libglew-dev \
            libopencv-dev \
            libsm-dev \
            libsuitesparse-dev \
            libxrender-dev \
            python3-dev \
            python3-pip \
            ros-$ROS_DISTRO-cv-bridge \
            ros-$ROS_DISTRO-eigen-conversions \
            ros-$ROS_DISTRO-pcl-conversions \
            ros-$ROS_DISTRO-pcl-ros \
            ros-$ROS_DISTRO-pluginlib \
            ros-$ROS_DISTRO-random-numbers \
            ros-$ROS_DISTRO-ros-base \
            ros-$ROS_DISTRO-rosbash \
            ros-$ROS_DISTRO-roslaunch \
            ros-$ROS_DISTRO-rviz \
            ros-$ROS_DISTRO-tf-conversions \
            ros-$ROS_DISTRO-cv-bridge \
            ros-$ROS_DISTRO-image-transport \
            ros-$ROS_DISTRO-eigen-conversions \
            rsync \
            stow \
            unzip \
            virtualenv \
    && \
    rm -rf /var/lib/apt/lists/*
