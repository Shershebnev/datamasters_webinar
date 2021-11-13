import os
import shutil
from argparse import ArgumentParser
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import LABEL2IDX, SEED


def get_paths(data_path: str) -> Tuple[List[str], List[str]]:
    """Get paths and labels for the dataset

    :param data_path: path to directory with data
    :return: list of paths and list of corresponding labels
    """
    paths = []
    class_labels = []
    for label in LABEL2IDX.keys():
        label_paths = [os.path.join(label, item) for item in os.listdir(os.path.join(data_path, label))]
        paths.extend(label_paths)
        class_labels.extend([label] * len(label_paths))
    return paths, class_labels


def split(paths: List[str], class_labels: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Splits data into train/val/test sets

    :param paths: list of paths
    :param class_labels: list of corresponding labels
    :return: train paths, val paths, test paths
    """
    train_paths, val_test_paths, _, val_test_labels = train_test_split(paths, class_labels, test_size=0.1,
                                                                       random_state=SEED)
    val_paths, test_paths, _, _ = train_test_split(val_test_paths, val_test_labels, test_size=0.5, random_state=SEED)
    return train_paths, val_paths, test_paths


def copy_files(data_path: str, paths: List[str], dataset_name: str, verbose: bool = False) -> None:
    """Copy files into data_path/dataset_name (train/val/test) directories

    :param data_path: path to original data
    :param paths: list of paths
    :param dataset_name: train/val/test
    :param verbose: show tqdm progress bar
    """
    for path in tqdm(paths, total=len(paths), desc=f"Copying {dataset_name} files", disable=not verbose):
        shutil.copy(os.path.join(data_path, path), os.path.join(data_path, dataset_name, path))


def main(data_path: str, verbose: int) -> None:
    """Prepare dataset - split into train/val/test, copy files into respective folder.
    Original structure of the dataset is expected to follow the following structure:
    args.data_path/
        class_1/
            image1.jpg
            image2.jpg
        ...
        class_2/
            image1.jpg
            image2.jpg
        ...

    Final structure of the dataset:
    args.data_path/
        train/
            class_1/
                image1.jpg
                image2.jpg
                ...
            ...
        val/
            class_1/
                image1.jpg
                image2.jpg
                ...
            ...
        test/
            class_1/
                image1.jpg
                image2.jpg
                ...
            ...

    :param data_path: path to original data
    :param verbose: show tqdm progress bar
    """
    paths, class_labels = get_paths(data_path)
    train_paths, val_paths, test_paths = split(paths, class_labels)

    for dataset_name in ["train", "test", "val"]:
        for label in LABEL2IDX.keys():
            os.makedirs(os.path.join(data_path, dataset_name, label), exist_ok=True)
    for ds_paths, dataset_name in zip([train_paths, val_paths, test_paths], ["train", "val", "test"]):
        copy_files(data_path, ds_paths, dataset_name, verbose != 0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", dest="data_path", help="Path to directory with data")
    parser.add_argument("--verbose", dest="verbose", help="Verbosity level: 0, 1, 2", type=int, choices=[0, 1, 2],
                        default=1)

    args = parser.parse_args()
    main(args.data_path, args.verbose)
