FROM ros:melodic

# PYTHON2
RUN apt-get update -y && apt-get install -y ros-melodic-compressed-image-transport
RUN apt-get install -y python-pip
RUN pip install matplotlib
RUN pip install gym

#PYTHON3
RUN apt-get install -y python3-pip
RUN pip3 install catkin_pkg
RUN pip3 install pyyaml
RUN pip3 install empy
RUN pip3 install opencv-python
RUN pip3 install gym
RUN pip3 install rospkg
RUN pip3 install tensorflow
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install seaborn



WORKDIR /root/projects/

# PYTHON2
# RUN echo 'catkin_make install && source /root/projects/devel/setup.bash' >> /root/.bashrc

# PYTHON3
RUN echo 'catkin_make install --cmake-args -DPYTHON_VERSION=3.6 && source /root/projects/devel/setup.bash' >> /root/.bashrc

