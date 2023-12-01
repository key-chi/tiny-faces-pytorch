FROM nvcr.io/nvidia/pytorch:22.10-py3

# Init
WORKDIR /home

ENV FORCE_CUDA=1
ENV MMCV_WITH_OPS=1

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
    build-essential \
    libatlas-base-dev \
    liblapack-dev \
    gfortran \
    libgl1-mesa-dev
RUN python3 -m pip install --upgrade pip setuptools wheel

# Define environment variable
#ENV NAME tiny-faces-pytorch

# Run the application when the container launches
#CMD ["/bin/bash"]