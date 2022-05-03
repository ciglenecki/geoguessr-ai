import os
from pathlib import Path
from fastapi import APIRouter
from app.config import config
from datamodule_geoguesser import GeoguesserDataModulePredict
from defaults import DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_SIZE
from model_classification import LitModelClassification, LitSingleModel
from model_regression import LitModelRegression
from utils_paths import PATH_DATA_COMPLETE, PATH_DATA_RAW, PATH_MODEL
from ..logger import logger
from fastapi import FastAPI, HTTPException
from app.server_store import server_store
from torchvision import transforms
import pytorch_lightning as pl
from predict import InferenceWriter
import torch

router = APIRouter()


@router.post("/model/{model_name}/predict", tags=["model"])
def read_model(model_name: str):
    model_name_path_dict = server_store.refresh_model_filepaths()
    if model_name not in model_name_path_dict:
        raise HTTPException(404, "Model {} not found.".format(model_name))
    model_path = model_name_path_dict[model_name]

    is_regression = False
    use_single_images = False
    model_constructor = (
        LitSingleModel if use_single_images else (LitModelRegression if is_regression else LitModelClassification)
    )

    model = model_constructor.load_from_checkpoint(str(model_path), batch_size=2)
    model.eval()

    mean, std = DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_SIZE

    image_transform = transforms.Compose(
        [
            transforms.Resize(model.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    predict_datamodule = GeoguesserDataModulePredict(
        [Path(PATH_DATA_RAW, "images", "test")],
        num_classes=model.num_classes,
        image_transform=image_transform,
    )
    with torch.no_grad():
        trainer = pl.Trainer(
            log_every_n_steps=1,
            callbacks=[InferenceWriter(Path("./here/df.csv"))],
            checkpoint_callback=False,
            logger=False,
        )
        predictions = trainer.predict(
            model=model,
            datamodule=predict_datamodule,
        )
    print(predictions)

    # model_path = Path(PATH_MODEL, model_name)
    # model = LitModelClassification()

    # # uses in_dim=32, out_dim=10
    # model = LitModel.load_from_checkpoint(PATH)

    # # uses in_dim=128, out_dim=10
    # model = LitModel.load_from_checkpoint(PATH, in_dim=128, out_dim=10)

    # logger.log(0, "test2")
    # logger.info("test")
    return {"username": "fakecurrentuser"}


@router.get("/models")
async def get_models():
    model_names = [model_filepath.stem for model_filepath in server_store.refresh_model_filepaths().values()]
    return {"item": model_names}
