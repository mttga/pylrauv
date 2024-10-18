FROM osrf/lrauv:latest

# Everything (ros2 and ros_gz) needs to be installed from source for compatibility with the Ubuntu and Gazebo versions

# FIRST STEP: Install ROS2 Rolling from source (bins are available only for ubuntu 22, here 20.04)
# following https://docs.ros.org/en/rolling/Installation/Alternatives/Ubuntu-Development-Setup.html

# get apt dependencies 
RUN sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    sudo apt update && sudo apt install -y \
    python3-flake8-docstrings \
    python3-pip \
    python3-pytest-cov \
    python3-lark-parser \
    ros-dev-tools

# get python dependencies
RUN python3 -m pip install -U \
    flake8-blind-except \
    flake8-builtins \
    flake8-class-newline \
    flake8-comprehensions \
    flake8-deprecated \
    flake8-import-order \
    flake8-quotes \
    "pytest>=5.3" \
    pytest-repeat \
    pytest-rerunfailures \
    netifaces \
    lark lark-parser

# get ros2 code
RUN sudo mkdir -p ~/ros2_rolling/src && \
    cd ~/ros2_rolling && \
    sudo vcs import --input https://raw.githubusercontent.com/ros2/ros2/rolling/ros2.repos src

# get rosdep dependencies
RUN sudo apt upgrade -y && \
    sudo rosdep init && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y --skip-keys "fastcdr rti-connext-dds-6.0.1 urdfdom_headers"

# additional libraries needed for buildig 
RUN sudo apt-get install -y libacl1-dev \
    libxrandr-dev \
    libxcursor-dev \
    libasio-dev \
    libtinyxml2-dev \
    libxaw7-dev \
    sip-dev pyqt5-dev python-sip-dev pyqt5-dev-tools

# ignore to build the demos
RUN sudo touch ~/ros2_rolling/src/ros2/demos/AMENT_IGNORE

# build 
RUN cd ~/ros2_rolling/ && \
    sudo colcon build --symlink-install

###########################################

# SECOND STEP: install my fork of ros_gz which is integrated in this repo and contains lrauv 
ENV GZ_VERSION garden

# Setup the ros gz_lrauv_workspace
ADD /ros_lrauv /home/developer/ros_lrauv

# install dependencies
RUN cd ~/ros_lrauv && \
    rosdep install -r --from-paths src -i -y --rosdistro rolling

# build and install into workspace
# colcon needs to be runned as sudo here in order to have permission to create folders
# therefore ros2 and gz need to be sourced in the sudo command
# avoid to build with test because are still not written for lrauv messages in ros_gz_bridge 
RUN cd ~/ros_lrauv && \
    sudo /bin/bash -c "source /home/developer/gz_ws/install/setup.bash && \
        source /home/developer/lrauv_ws/install/setup.bash && \
        source /home/developer/ros2_rolling/install/setup.bash && \
        colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF"

# activate by default the packages in every new bash shell
RUN echo "source ~/gz_ws/install/setup.sh" >> /home/developer/.bashrc && \
    echo "source ~/lrauv_ws/install/setup.sh" >> /home/developer/.bashrc && \
    echo "source ~/ros2_rolling/install/setup.sh" >> /home/developer/.bashrc && \
    echo "source ~/ros_lrauv/install/setup.sh" >> /home/developer/.bashrc

# STEP 3: additional python packages
RUN python3 -m pip install -U pyproj