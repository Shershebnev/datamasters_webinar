import os
from argparse import ArgumentParser

import wandb
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # pylint: disable=E0611
from wandb.keras import WandbCallback

from config import AVAILABLE_MODELS, MODEL_DIR, NUM_CLASSES
from utils import get_dataset, get_latest_version, get_model


def main(batch_size: int, image_shape: int, model_type: str, epochs: int, data_path: str, verbose: int) -> None:
    """Train network

    :param batch_size: batch size
    :param image_shape: image width(==height) for input images
    :param model_type: name of the model architecture, e.g. resnet18 or vgg16. For full list see
                       classification_models.tfkeras.Classifiers.models_names()
    :param epochs: number of epochs
    :param data_path: path to data
    :param verbose: verbosity level
    """
    run = wandb.init(project="wandb-demo", entity="shershebnev", tags=["training"])
    run.use_artifact('shershebnev/wandb-demo/data:v0', type='data')
    cfg = wandb.config
    cfg.update({"epochs": epochs, "batch_size": batch_size, "model_type": model_type, "image_shape": image_shape,
                "data_path": data_path})
    train_ds = get_dataset(os.path.join(data_path, "train"), batch_size, image_shape, model_type)
    val_ds = get_dataset(os.path.join(data_path, "val"), batch_size, image_shape, model_type, apply_aug=False)

    model_latest_version = get_latest_version(model_type)
    callbacks = [ModelCheckpoint(f"{MODEL_DIR}/{model_type}/{model_latest_version + 1}", monitor="val_loss",
                                 save_best_only=True),
                 EarlyStopping(monitor="val_loss", patience=7, verbose=verbose), WandbCallback()]
    model = get_model(model_type, image_shape, NUM_CLASSES)
    model.compile(optimizer="SGD", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, verbose=verbose)
    model_data = wandb.Artifact("model", type="model")
    model_data.add_dir(f"{MODEL_DIR}/{model_type}/{model_latest_version + 1}")
    run.log_artifact(model_data)
    run.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", dest="batch_size", help="Batch size", type=int)
    parser.add_argument("--image_shape", dest="image_shape", help="Model input image shape, integer", type=int)
    parser.add_argument("--model_type", dest="model_type", help=f"Model type, one of: {', '.join(AVAILABLE_MODELS)}",
                        choices=AVAILABLE_MODELS)
    parser.add_argument("--epochs", dest="epochs", help="Number of epochs", type=int)
    parser.add_argument("--data_path", dest="data_path", help="Path to directory with data")
    parser.add_argument("--verbose", dest="verbose", help="Verbosity level: 0, 1, 2", type=int, choices=[0, 1, 2],
                        default=1)

    args = parser.parse_args()
    main(args.batch_size, args.image_shape, args.model_type, args.epochs, args.data_path, args.verbose)
