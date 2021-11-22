import cv2
import numpy as np
import pika
from alibi_detect.utils.saving import load_detector
from prometheus_client import Counter, start_http_server
from tqdm import tqdm

from drift_config import TEST_SIZE, IMAGE_SHAPE, QUEUE_NAME, RABBIT_HOST_NAME


drift_detector = load_detector("models/ksdrift")

connection = pika.BlockingConnection(pika.ConnectionParameters(RABBIT_HOST_NAME))
channel = connection.channel()
consumer = channel.consume(queue=QUEUE_NAME)

counters = {i: Counter(f"drift_{bool(i)}", "Whether found drift in data") for i in range(0, 2)}
start_http_server(8000)

while True:
    sample = np.zeros((TEST_SIZE, IMAGE_SHAPE, IMAGE_SHAPE, 3))
    for idx in tqdm(range(TEST_SIZE)):
        img_b = next(consumer)[2]
        nparr = np.fromstring(img_b, np.uint8)
        img = cv2.resize(cv2.imdecode(nparr, cv2.IMREAD_COLOR), (IMAGE_SHAPE, IMAGE_SHAPE))
        sample[idx] = img
    is_drift = drift_detector.predict(sample)["data"]["is_drift"]
    counters[is_drift].inc()
