import glob
import random
import requests

import cv2
import numpy as np
from tqdm import tqdm


np.random.seed(42)
URL = "http://localhost:5000/predict"


def gauss_noise(image: np.ndarray, mean: float = 0.0, var: float = 0.001) -> np.ndarray:
    """Add gaussian noise to an image

    :param image: input image
    :param mean: mean of distribution to sample noise from
    :param var: variance of distribution to sample noise from
    :return: image with added noise
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def corrupt(image: np.ndarray, prob: float) -> np.ndarray:
    """Corrupt an image with prob chance

    :param image: input image
    :param prob: probability to corrupt an image with noise
    :return: corrupted or original image
    """
    if random.random() < prob:
        return gauss_noise(image)
    else:
        return image


files = glob.glob("data/test/*/*JPG")
content_type = 'image/jpeg'
headers = {'content-type': content_type}
for i in tqdm(range(0, 101, 20), total=6):
    prob_corrupt = i / 100
    for i in tqdm(range(len(files)), total=len(files)):
        img = cv2.imread(files[i])
        img = corrupt(img, prob_corrupt)
        data = cv2.imencode('.jpg', img)[1].tostring()
        response = requests.post(URL, data=data, headers=headers)
