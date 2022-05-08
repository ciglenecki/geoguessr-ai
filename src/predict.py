from __future__ import annotations, division, print_function

from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchvision import transforms

from datamodule_geoguesser import GeoguesserDataModulePredict
from config import DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD
from model_classification import LitModelClassification, LitSingleModel
from model_regression import LitModelRegression
from utils_paths import PATH_DATA_ORIGINAL


class InferenceWriter(Callback):
    def __init__(self, output_path: Path):
        super().__init__()
        self.output_path = output_path
        self.df = pd.DataFrame(columns=["uuid", "latitude", "longitude"])
        self.df.to_csv(self.output_path, mode="w+", index=False, header=True)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        tmp_df = pd.DataFrame(outputs)
        tmp_df.to_csv(self.output_path, mode="a+", index=False, header=False)


if __name__ == "__main__":
    is_regression = False
    use_single_images = False
    model_constructor = (
        LitSingleModel if use_single_images else (LitModelRegression if is_regression else LitModelClassification)
    )
    ckpt_path = str(
        Path(
            "reports/Papa_86__num_classes_53__05-01-20-33-41/version_0/checkpoints/resnext101_32x8d__haversine_0.0133__val_acc_0.00__val_loss_228751149367296.00.ckpt"
        )
    )
    model = model_constructor.load_from_checkpoint(ckpt_path, strict=False, batch_size=2)
    model.eval()

    mean, std = DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD

    image_transform_train = transforms.Compose(
        [
            transforms.Resize(model.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    predict_datamodule = GeoguesserDataModulePredict([Path(PATH_DATA_ORIGINAL, "test")], num_classes=model.num_classes)

    trainer = pl.Trainer(
        log_every_n_steps=1, callbacks=[InferenceWriter(Path("./here/df.csv"))], checkpoint_callback=False, logger=False
    )
    predictions = trainer.predict(model=model, datamodule=predict_datamodule)
    print(predictions)
