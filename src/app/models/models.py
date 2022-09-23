from typing import List, Optional
from uuid import UUID

from fastapi import UploadFile
from pydantic import BaseModel


class PostPredictDatasetRequest(BaseModel):
    dataset_directory_path: str
    csv_filename: Optional[str] = None


class ModelLatLng(BaseModel):
    latitude: float
    longitude: float


class PredictImagesResponse(ModelLatLng):
    pass


class PredictImagesCardinalResponse(ModelLatLng):
    pass


class PredictDirectoryReponse(BaseModel):
    uuid: UUID
    latitude: float
    longitude: float


class PostPredictCardinal(list[UploadFile]):
    pass
