from typing import Optional
from uuid import UUID

from fastapi import UploadFile
from pydantic import BaseModel


class PostPredictDatasetRequest(BaseModel):
    dataset_directory_path: str
    csv_path: Optional[str] = None


class ModelLatLng(BaseModel):
    latitude: float
    longitude: float


class PredictImagesResponse(BaseModel):
    uuid: ModelLatLng


class PredictDirectoryReponse(BaseModel):
    latitude: float
    longitude: float
    uuid: UUID


class PostPredictCardinal(BaseModel):
    north: UploadFile
    east: UploadFile
    south: UploadFile
    west: UploadFile
