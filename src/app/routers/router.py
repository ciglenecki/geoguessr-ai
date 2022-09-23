from typing import List

from fastapi import APIRouter, UploadFile

import app.controller.controller as controller
import app.models.models as models
from app.descriptions import (
    get_models_desc,
    predict_cardinal_desc,
    predict_directory_desc,
    predict_images_desc,
)

router = APIRouter(prefix="/model")


@router.get(
    "s",
    tags=["available models"],
    response_model=list[str],
    description=get_models_desc,
)
async def get_models():
    return controller.get_models()


@router.post(
    "/{model_name}/predict-images",
    tags=["predict"],
    response_model=list[models.PredictImagesResponse],
    description=predict_images_desc,
)
async def predict_images(model_name: str, images: list[UploadFile]):
    return await controller.predict_images(model_name, images)


@router.post(
    "/{model_name}/predict-cardinal-images",
    tags=["predict"],
    response_model=list[models.PredictImagesCardinalResponse],
    description=predict_cardinal_desc,
)
async def predict_cardinal_images(model_name: str, images: list[UploadFile]):
    return await controller.predict_cardinal_images(model_name, images)


@router.post(
    "/{model_name}/predict-directory",
    tags=["predict"],
    response_model=list[models.PredictDirectoryReponse],
    description=predict_directory_desc,
)
def predict_dataset(model_name: str, body: models.PostPredictDatasetRequest):
    return controller.predict_dataset(model_name, body)
