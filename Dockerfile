FROM nvcr.io/nvidia/tensorflow:21.02-tf2-py3
COPY requirements.txt /opt
RUN pip3 install -r /opt/requirements.txt
COPY . /opt
