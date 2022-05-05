from typing import List

from fastapi import APIRouter, UploadFile
from app.descriptions import get_models_desc, predict_images_desc, predict_cardinal_desc, predict_directory_desc
import app.models.models as models
import app.controller.controller as controller

router = APIRouter(prefix="/model")


@router.get("s", tags=["available models"], response_model=List[str], description=get_models_desc)
async def get_models():
    return controller.get_models()


@router.post(
    "/{model_name}/predict-images",
    tags=["predict"],
    response_model=List[models.PredictImagesResponse],
    description=predict_images_desc,
)
async def predict_images(model_name: str, images: List[UploadFile]):
    return await controller.predict_images(model_name, images)


@router.post(
    "/{model_name}/predict-cardinal-images",
    tags=["predict"],
    response_model=List[models.PredictImagesResponse],
    description=predict_cardinal_desc,
)
async def predict_cardinal_images(model_name: str, images: List[models.PostPredictCardinal]):
    return await controller.predict_cardinal_images(model_name, images)


@router.post(
    "/{model_name}/predict-directory",
    tags=["predict"],
    response_model=List[models.PredictDirectoryReponse],
    description=predict_directory_desc,
)
def read_model(model_name: str, body: models.PostPredictDatasetRequest):
    return controller.read_model(model_name, body)
