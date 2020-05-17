FROM nvidia/cuda:10.0-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
    libopencv-dev \
    git \
    wget

RUN cd /opt && \
    git clone https://github.com/AlexeyAB/darknet.git && \
    cd darknet && \
    git checkout 2652263727a7036dfb587fbda98bd30181d8a616 && \
    sed -i -e "/GPU=/s/0/1/" Makefile && \
    sed -i -e "/CUDNN=/s/0/1/" Makefile && \
    sed -i -e "/OPENCV=/s/0/1/" Makefile && \
    make -j$(nproc)

WORKDIR /opt/darknet
