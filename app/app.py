import json
import logging

import cv2
import numpy as np
import pika
from flask import Flask, request, Response
from prometheus_flask_exporter import Counter, PrometheusMetrics

from app.flask_config import MODEL_PATH, IMAGE_SHAPE, RABBIT_HOST_NAME, QUEUE_NAME
from app.model import Model
from config import IDX2LABEL


app = Flask(__name__)
logger = logging.getLogger(__name__)
app.logger.handlers.extend(logger.handlers)
app.logger.setLevel(logging.DEBUG)
app.logger.debug("Starting")
metrics = PrometheusMetrics(app)
model = Model(MODEL_PATH, IMAGE_SHAPE)
metrics.info("app_info", "App Info, this can be anything you want", version="1.0.0")
counters = {i: Counter(f"label_{i}", "Count of specific label predictions") for i in range(len(IDX2LABEL))}

connection = pika.BlockingConnection(pika.ConnectionParameters(RABBIT_HOST_NAME))
channel = connection.channel()
channel.queue_declare(queue=QUEUE_NAME)


@metrics.gauge("predict_timing", "Single prediction time")
def _get_prediction(img):
    return model.predict(img)


@app.route("/predict", methods=["POST"])
@metrics.counter("predict_call_count", "Number of predict requests")
def predict():
    r = request
    channel.basic_publish(exchange='', routing_key=QUEUE_NAME, body=r.data)
    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    prediction = _get_prediction(img)
    response = {"label": prediction}
    counters[prediction].inc()
    app.logger.debug(f"Predicted {prediction}")
    return Response(response=json.dumps(response), status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
