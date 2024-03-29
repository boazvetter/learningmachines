FROM ros:melodic

RUN apt-get update -y && apt-get install -y ros-melodic-compressed-image-transport
RUN apt-get install -y python-pip
RUN pip install torch
RUN pip install matplotlib
RUN pip install gym

WORKDIR /root/projects/
RUN echo 'catkin_make install && source /root/projects/devel/setup.bash' >> /root/.bashrc

