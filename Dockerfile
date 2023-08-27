FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04
USER root

WORKDIR /root/src
RUN mkdir -p /root/src
COPY requirements.txt /root/src

RUN apt-get update && apt-get install -y python3-pip git
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt