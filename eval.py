import os
from argparse import ArgumentParser

import wandb
from tensorflow.keras.models import load_model  # pylint: disable=E0611
from wandb.keras import WandbCallback

from config import AVAILABLE_MODELS
from utils import get_dataset


def main(batch_size: int, image_shape: int, weights_path: str, data_path: str, model_type: str, verbose: int) -> None:
    """Evaluate trained network on test set

    :param batch_size: batch size
    :param image_shape: image width(==height) for input images
    :param weights_path: path to saved model weights file (.h5)
    :param data_path: path to data
    :param model_type: name of the model architecture, e.g. resnet18 or vgg16. For full list see
                       classification_models.tfkeras.Classifiers.models_names()
    :param verbose: verbosity level
    """
    run = wandb.init(project="wandb-demo", entity="shershebnev", tags=["evaluation"])
    run.use_artifact('shershebnev/wandb-demo/data:v0', type='data')
    test_ds = get_dataset(os.path.join(data_path, "test"), batch_size, image_shape, model_type=model_type,
                          shuffle=False, apply_aug=False)
    model = load_model(weights_path)
    model.evaluate(test_ds, verbose=verbose, callbacks=[WandbCallback()])
    run.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", dest="batch_size", help="Batch size", type=int)
    parser.add_argument("--image_shape", dest="image_shape", help="Model input image shape, integer", type=int)
    parser.add_argument("--weights_path", dest="weights_path", help="Path to model weights")
    parser.add_argument("--data_path", dest="data_path", help="Path to directory with data")
    parser.add_argument("--model_type", dest="model_type", choices=AVAILABLE_MODELS,
                        help=f"Model type used, one of: {', '.join(AVAILABLE_MODELS)}, for preprocessing purpose")
    parser.add_argument("--verbose", dest="verbose", help="Verbosity level: 0, 1, 2", type=int, choices=[0, 1, 2],
                        default=1)

    args = parser.parse_args()
    main(args.batch_size, args.image_shape, args.weights_path, args.data_path, args.model_type, args.verbose)
