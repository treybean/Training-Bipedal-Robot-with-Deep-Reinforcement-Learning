FROM ubuntu:18.04

RUN \
  apt -y update && \
  apt install -y keyboard-configuration && \
  apt install -y \ 
  python3-setuptools \
  python3-pip \
  python3-dev \
  python-pyglet \
  python3-opengl \
  python-box2d \
  libjpeg-dev \
  libboost-all-dev \
  libsdl2-dev \
  libosmesa6-dev \
  patchelf \
  ffmpeg \
  xvfb \
  wget \
  unzip && \
  apt clean && \
  rm -rf /var/lib/apt/lists/* && \
  pip3 install tensorflow \
  keras \
  box2d-py \
  gym