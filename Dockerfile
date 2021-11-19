FROM nvcr.io/nvidia/tensorflow:21.02-tf2-py3
COPY requirements.txt /opt
RUN pip3 install -r /opt/requirements.txt
COPY . /opt
WORKDIR /opt
CMD gunicorn -w 1 --bind 0.0.0.0:5000 app.app:app
