import glob
import requests

import cv2
import numpy as np
from classification_models.tfkeras import Classifiers
from tensorflow.keras.models import load_model
from tqdm import tqdm

from config import IDX2LABEL


URL = "http://localhost:8003/seldon/seldon-system/resnet18"


def predict(data):
    data = {
        "inputs": [
            {
                "name": "data",
                "data": data.tolist(),
                "datatype": "FP32",
                "shape": data.shape,
            }
        ]
    }
    r = requests.post(f"{URL}/v2/models/resnet18/infer", json=data)
    predictions = np.array(r.json()["outputs"][0]["data"]).reshape(r.json()["outputs"][0]["shape"])
    return predictions


# read files and prepare batch
files = glob.glob("data/test/*/*JPG")
SUBSET_SIZE = min(len(files), 5)
test_subset = files[:SUBSET_SIZE]
payload = np.zeros((SUBSET_SIZE, 224, 224, 3))
preprocess_input = Classifiers.get("resnet18")[1]
for idx, file in tqdm(enumerate(test_subset), total=SUBSET_SIZE):
    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    img = preprocess_input(cv2.resize(img, (224, 224)))
    payload[idx] = img

# predictions from the deployed model
predictions_np = predict(payload)

# predictions from same model stored locally
model = load_model("models/resnet18/1/model.savedmodel")
local_predictions_np = model.predict(payload)

# check labels and actual softmax values are the same
for pred, local_pred, file in zip(np.argmax(predictions_np, axis=1), np.argmax(local_predictions_np, axis=1), test_subset):
    gt = file.split("/")[2]
    print(f"Predicted {IDX2LABEL[pred]}, locally predicted {IDX2LABEL[local_pred]}, actual {gt} for {file}")

print(np.allclose(predictions_np, local_predictions_np))
