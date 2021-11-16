import os
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from classification_models.tfkeras import Classifiers
from tensorflow.data import AUTOTUNE  # pylint: disable=E0401
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # pylint: disable=E0611
from tensorflow.keras.models import Model  # pylint: disable=E0611
from tensorflow.keras.preprocessing import image_dataset_from_directory  # pylint: disable=E0611
from tensorflow.python.keras.engine.functional import Functional  # typing  # pylint: disable=E0611
from tensorflow.python.data.ops.dataset_ops import BatchDataset  # typing  # pylint: disable=E0611

from config import AUG, MODEL_DIR, SEED


def get_latest_version(model_type: str) -> int:
    """Get latest model version from model dir for specified model type. In real life this should consult some remote
    storage like S3 (directly or through, e.g. wandb), not local directory. But good enough for the demo purposes :)

    :param model_type: model type
    :return: next model version
    """
    versions = os.listdir(f"{MODEL_DIR}/{model_type}")
    if not versions:  # no model of model_type logged
        return 0
    return int(sorted(versions, key=lambda x: int(x))[-1])


def get_model(model_type: str, image_shape: int, num_classes: int) -> Functional:
    """Returns specified model architecture pretrained on imagenet and ready for finetuning

    :param model_type: name of the model architecture, e.g. resnet18 or vgg16. For full list see
                       classification_models.tfkeras.Classifiers.models_names()
    :param image_shape: image width(==height) for input images
    :param num_classes: number of classes
    :return: Model
    """
    model, _ = Classifiers.get(model_type)
    base_model = model(input_shape=(image_shape, image_shape, 3), weights="imagenet", include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=[base_model.input], outputs=[output])
    return model


def _aug_fn(image: np.ndarray) -> np.ndarray:
    """Augmentation function

    :param image: image to augment
    :return: augmented image
    """
    return AUG(image=image)["image"]


def _preprocessing(image: np.ndarray, label: int, preprocess_fn: Callable, apply_aug: bool) -> Tuple[np.ndarray, int]:
    """Image preprocessing function

    :param image: image
    :param label: label
    :param preprocess_fn: preprocessing function
    :param apply_aug: whether to apply augmentations
    :return: preprocessed and (optionally) augmented image
    """
    if apply_aug:
        return preprocess_fn(tf.numpy_function(func=_aug_fn, inp=[image], Tout=tf.float32)), label
    return preprocess_fn(image), label


def get_dataset(path: str, batch_size: int, image_shape: int, model_type: str, labels: Optional[str] = "inferred",
                label_mode: Optional[str] = "categorical", shuffle: bool = True, apply_aug: bool = True) -> \
        BatchDataset:
    """Returns tf.data.Dataset object

    :param path: path to directory with data. For train/val/test sets see
                 https://keras.io/api/preprocessing/image/#image_dataset_from_directory-function for directory structure
    :param batch_size: batch size
    :param image_shape: image width(==height) for input images
    :param model_type: name of the model architecture, e.g. resnet18 or vgg16. For full list see
                       classification_models.tfkeras.Classifiers.models_names()
    :param labels: list of labels, None or inferred (see link above)
    :param label_mode: labels formatting method - int, categorical, binary or None (see link above)
    :param shuffle: whether to shuffle data
    :param apply_aug: whether to apply augmentations
    :return: Dataset object
    """
    preprocess_input = Classifiers.get(model_type)[1]
    if labels:
        return image_dataset_from_directory(directory=path, labels=labels, label_mode=label_mode, batch_size=batch_size,
                                            image_size=(image_shape, image_shape), shuffle=shuffle, seed=SEED).map(
            partial(_preprocessing, preprocess_fn=preprocess_input, apply_aug=apply_aug), num_parallel_calls=AUTOTUNE,
            deterministic=True)
    return image_dataset_from_directory(directory=path, labels=labels, label_mode=label_mode, batch_size=batch_size,
                                        image_size=(image_shape, image_shape), shuffle=shuffle, seed=SEED).map(
        preprocess_input, num_parallel_calls=AUTOTUNE, deterministic=True)


def _prepare_image(path: str, image_shape: int) -> np.ndarray:
    """Loads and preprocesses single image

    :param path: path to an image
    :param image_shape: image width(==height) for input images
    :return: processed image
    """
    return cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (image_shape, image_shape))


def _prepare_batches_from_dir(path: str, batch_size: int, image_shape: int, model_type: str) \
        -> Tuple[BatchDataset, List[str]]:
    """Returns tf.data.Dataset object (without labels) for data @ path

    :param path: path to directory with images
    :param batch_size: batch size
    :param image_shape: image width(==height) for input images
    :param model_type: name of the model architecture, e.g. resnet18 or vgg16. For full list see
                       classification_models.tfkeras.Classifiers.models_names()
    :return: Dataset object and list of file paths
    """
    dataset = get_dataset(path, batch_size, image_shape, labels=None, label_mode=None, model_type=model_type,
                          shuffle=False, apply_aug=False)
    return dataset, dataset._input_dataset.file_paths


def _prepare_batch_from_image(path: str, image_shape: int) -> Tuple[np.ndarray, List[str]]:
    """Converts image into a batch

    :param path: path to an image
    :param image_shape: image width(==height) for input images
    :return: 4D numpy array and list of file paths
    """
    return np.expand_dims(_prepare_image(path, image_shape), 0), [path]  # type: ignore


def prepare_batches(path: str, batch_size: int, image_shape: int, model_type: str) \
        -> Tuple[Union[BatchDataset, np.ndarray], List[str]]:
    """Prepares data @ path for use in predict method. If path is directory - creates tf.data.Dataset object;
    if path is file - creates 4D numpy array

    :param path: path to either an image or directory
    :param batch_size: batch size
    :param image_shape: image width(==height) for input images
    :param model_type: name of the model architecture, e.g. resnet18 or vgg16. For full list see
                       classification_models.tfkeras.Classifiers.models_names()
    :return: either 4D numpy array or Dataset object and list with file paths
    """
    if os.path.isdir(path):
        return _prepare_batches_from_dir(path, batch_size, image_shape, model_type)
    return _prepare_batch_from_image(path, image_shape)
