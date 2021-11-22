import os.path as path
import sys
import glob
import random
from argparse import ArgumentParser
from functools import partial
from inspect import getsourcefile
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

import cv2
import numpy as np
from alibi_detect.cd import KSDrift
from alibi_detect.cd.tensorflow import preprocess_drift
from alibi_detect.utils.saving import save_detector
from tqdm import tqdm

from drift_training_config import ENCODING_DIM, IMAGE_SHAPE
from utils import get_model


def get_training_data(data_path_pattern: str) -> np.ndarray:
    """Load training data into numpy array. NOTE: this might cause a memory problem with large datasets and/or small RAM

    :param data_path_pattern: pattern string which will be used to find training files, e.g. /data/train/*/*jpg
    :return: numpy array with loaded images
    """
    train_files = glob.glob(data_path_pattern)
    random.shuffle(train_files)
    x_ref = np.zeros((len(train_files), IMAGE_SHAPE, IMAGE_SHAPE, 3))
    for idx in tqdm(range(len(train_files)), total=len(train_files)):
        f = train_files[idx]
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SHAPE, IMAGE_SHAPE))
        x_ref[idx] = img
    return x_ref


def train_and_save(training_data: np.ndarray, model_save_path: str) -> None:
    """Train and save drift detector

    :param training_data: training data
    :param model_save_path: save path
    """
    encoder_net = get_model("vgg16", 224, ENCODING_DIM)
    preprocess_fn = partial(preprocess_drift, model=encoder_net, batch_size=128)
    cd = KSDrift(training_data, p_val=0.05, preprocess_fn=preprocess_fn)
    save_detector(cd, model_save_path)


def main(data_path_pattern: str, model_save_path: str) -> None:
    """Main function - prepares data, trains drift detector and saves it

    :param data_path_pattern: pattern string which will be used to find training files, e.g. /data/train/*/*jpg
    :param model_save_path: save path
    """
    training_data = get_training_data(data_path_pattern)
    train_and_save(training_data, model_save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path_pattern", help="Pattern for training data, e.g. /data/train/*/*JPG")
    parser.add_argument("--model_save_path", help="Path to output directory where model will be saved")
    args = parser.parse_args()
    main(args.data_path_pattern, args.model_save_path)
