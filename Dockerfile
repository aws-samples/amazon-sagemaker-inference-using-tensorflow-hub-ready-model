FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu18.04

RUN apt-get update \
    && apt-get install -y software-properties-common wget curl \
    && add-apt-repository -y ppa:jonathonf/python-3.8 \
    && apt-get install -y python3.8 python3.8-distutils

RUN ln -s /usr/bin/python3.8 /usr/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py

WORKDIR /opt/ml/code

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY ./inference.py /opt/ml/code/

ENTRYPOINT [ "python" , "inference.py" ]