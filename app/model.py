import cv2
import numpy as np
from tensorflow.keras.models import load_model


class Model:
    def __init__(self, model_path: str, shape: int) -> None:
        self.model = load_model(model_path)
        self.shape = shape

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        return np.expand_dims(cv2.resize(image, (self.shape, self.shape)), axis=0)

    def predict(self, image: np.ndarray) -> int:
        return int(np.argmax(self.model.predict(self._prepare_image(image))))
