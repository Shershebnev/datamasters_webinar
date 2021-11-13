from argparse import ArgumentParser

import numpy as np
from tensorflow.keras.models import load_model  # pylint: disable=E0611
from tqdm import tqdm

from config import AVAILABLE_MODELS, IDX2LABEL
from utils import prepare_batches


def main(batch_size: int, image_shape: int, weights_path: str, data_path: str, output_file: str, model_type: str,
         verbose: int) -> None:
    """Get predictions from the model for one or multiple images

    :param batch_size: batch size
    :param image_shape: image width(==height) for input images
    :param weights_path: path to saved model weights file (.h5)
    :param data_path: path to data
    :param output_file: path to file where predictions will be saved in CSV format
    :param model_type: name of the model architecture, e.g. resnet18 or vgg16. For full list see
                       classification_models.tfkeras.Classifiers.models_names()
    :param verbose: verbosity level
    """
    model = load_model(weights_path)
    batch, paths = prepare_batches(data_path, batch_size, image_shape, model_type)
    pred = model.predict(batch, verbose=verbose)
    with open(output_file, "w", ) as f:  # pylint: disable=W1514
        for idx, path in tqdm(enumerate(paths)):
            label = IDX2LABEL[int(np.argmax(pred[idx]))]
            f.write(f"{path},{label}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", dest="batch_size", help="Batch size", type=int, default=1)
    parser.add_argument("--image_shape", dest="image_shape", help="Model input image shape, integer", type=int)
    parser.add_argument("--weights_path", dest="weights_path", help="Path to model weights")
    parser.add_argument("--model_type", dest="model_type", choices=AVAILABLE_MODELS,
                        help=f"Model type used, one of: {', '.join(AVAILABLE_MODELS)}, for preprocessing purpose")
    parser.add_argument("--data_path", dest="data_path", help="Path either to directory with data or to image")
    parser.add_argument("--output_file", dest="output_file", help="Path to csv file where predictions will be saved")
    parser.add_argument("--verbose", dest="verbose", help="Verbosity level: 0, 1, 2", type=int, choices=[0, 1, 2],
                        default=1)

    args = parser.parse_args()
    main(args.batch_size, args.image_shape, args.weights_path, args.data_path, args.output_file, args.model_type,
         args.verbose)
