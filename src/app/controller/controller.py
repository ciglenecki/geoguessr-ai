import sys

sys.path.append("..")

import copy
import io
import os
from pathlib import Path
from typing import Dict, List, Literal

import pytorch_lightning as pl
import torch
from datamodule_geoguesser import GeoguesserDataModulePredict
from defaults import DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD
from fastapi import HTTPException, UploadFile, status
from model_classification import LitModelClassification
from PIL import Image
from predict import InferenceWriter
import app.models.models as models

from torchvision import transforms
from utils_model import crs_coords_to_degree, crs_coords_weighed_mean

from app.server_store import server_store


def _verify_model_name(model_name: str):
    model_name_path_dict = server_store.refresh_model_filepaths()
    if model_name not in model_name_path_dict:
        raise HTTPException(404, "Model '{}' not found.".format(model_name))


def _verify_dataset_directory_path(path_arg: str):
    dataset_directory_path = Path(path_arg).resolve()
    if not Path(dataset_directory_path).resolve().is_dir():
        raise HTTPException(404, "Dataset directory '{}' not found.".format(dataset_directory_path))
    return dataset_directory_path


def _load_model(model_name: str):
    with torch.no_grad():
        if model_name in server_store.cached_models:
            return server_store.cached_models[model_name]
        else:
            model_path = server_store.model_filepaths[model_name]
            # is_regression = False
            # use_single_images = False
            model_constructor = LitModelClassification
            model = model_constructor.load_from_checkpoint(str(model_path), batch_size=server_store.batch_size)
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            server_store.cached_models[model_name] = copy.deepcopy(model)
            return model


def _get_lat_lng_predictions(y_pred: torch.Tensor, model: LitModelClassification):
    pred_crs_coord = crs_coords_weighed_mean(y_pred, model.class_to_crs_weighted_map, top_k=5)
    pred_crs_coord = pred_crs_coord.cpu()
    pred_crs_coord_transformed = model.crs_scaler.inverse_transform(pred_crs_coord)
    pred_degree_coords = crs_coords_to_degree(pred_crs_coord_transformed)
    return pred_degree_coords


def _get_image_transform(image_size: int):
    mean, std = DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _process_image(image: Image.Image, image_size: int) -> torch.Tensor:
    image_transform = _get_image_transform(image_size)
    return image_transform(image)


async def predict_cardinal_images(model_name: str, images: List[models.PostPredictCardinal]):
    sides = images[0].dict().keys()
    print(sides)

    for image in images:
        for side in sides:
            print(image[side].content_type)
    exit(1)

    if any([image.content_type != "image/jpeg" and image.content_type != "image/png" for image in images]):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File has to be .jpg or .pong image")

    sides = ["0", "90", "180", "270"]

    all([Path(image.filename).stem for image in images])

    _verify_model_name(model_name)
    model = _load_model(model_name)

    response_object: List[Dict[Literal["latitude", "longitude"], float]] = []
    for image in images:
        image_io = await image.read()
        image_pil = Image.open(io.BytesIO(image_io))  # type: ignore

        image_tensor = _process_image(image_pil, model.image_size)
        image_tensor = image_tensor.unsqueeze(dim=0)
        image_list = [image_tensor] * 4

        y_pred = model.forward(image_list)
        pred_degree_coords = _get_lat_lng_predictions(y_pred, model)
        lat, lng = pred_degree_coords.squeeze()
        lat, lng = float(lat), float(lng)
        response_object.append({"latitude": lat, "longitude": lng})
    return response_object


async def predict_images(model_name: str, images: List[UploadFile]):

    if any([image.content_type != "image/jpeg" and image.content_type != "image/png" for image in images]):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File has to be .jpg or .pong image")

    _verify_model_name(model_name)
    model = _load_model(model_name)

    response_object: List[Dict[Literal["latitude", "longitude"], float]] = []
    for image in images:
        image_io = await image.read()
        image_pil = Image.open(io.BytesIO(image_io))  # type: ignore

        image_tensor = _process_image(image_pil, model.image_size)
        image_tensor = image_tensor.unsqueeze(dim=0)
        image_list = [image_tensor] * 4

        y_pred = model.forward(image_list)
        pred_degree_coords = _get_lat_lng_predictions(y_pred, model)
        lat, lng = pred_degree_coords.squeeze()
        lat, lng = float(lat), float(lng)
        response_object.append({"latitude": lat, "longitude": lng})
    return response_object


def get_models():
    model_names = [model_filepath.stem for model_filepath in server_store.refresh_model_filepaths().values()]
    return model_names


def read_model(model_name: str, body: models.PostPredictDatasetRequest):

    _verify_model_name(model_name)
    dataset_directory_path = _verify_dataset_directory_path(body.dataset_directory_path)

    if body.csv_path:
        os.makedirs(os.path.dirname(body.csv_path), exist_ok=True)
        open(body.csv_path, "w").close()

    model = _load_model(model_name)
    with torch.no_grad():

        image_transform = _get_image_transform(model.image_size)

        predict_datamodule = GeoguesserDataModulePredict(
            [dataset_directory_path],
            num_classes=model.num_classes,
            image_transform=image_transform,
            dataset_frac=1,
            batch_size=server_store.batch_size,
        )
        predict_datamodule.setup()

        callbacks = []
        if body.csv_path:
            callbacks.append(InferenceWriter(Path(body.csv_path)))

        trainer = pl.Trainer(
            callbacks=callbacks,
            checkpoint_callback=False,
            logger=False,
        )
        predictions = trainer.predict(
            model=model,
            datamodule=predict_datamodule,
        )  # type: ignore [lightning doesnt know what we return in predict]

    predictions: List[Dict]  # keys in one dictionary: uuid longitude latitude
    response_dict: Dict[str, Dict[Literal["latitude", "longitude"], float]] = {}
    for pred_dict in predictions:
        for uuid, longitude, latitude in zip(pred_dict["uuid"], pred_dict["longitude"], pred_dict["latitude"]):
            response_dict[uuid] = {"longitude": longitude, "latitude": latitude}
    return response_dict
