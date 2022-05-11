from __future__ import annotations, division, print_function
import argparse
import os

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
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    lightning_parser = pl.Trainer.add_argparse_args(parser)
    lightning_parser.set_defaults()

    user_group = parser.add_argument_group("my")

    user_group.add_argument(
        "--checkpoint",
        help="Path to the model checkpoint",
        type=str,
        required=True,
    )
    user_group.add_argument(
        "--images-dir",
        help="Path to images directory",
        type=str,
        required=True,
    )
    user_group.add_argument(
        "--out",
        help="Path to csv output",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    """Separate Namespace into two Namespaces"""
    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        if group.title:
            args_dict[group.title] = argparse.Namespace(**group_dict)

    args, pl_args = args_dict["my"], args_dict["pl.Trainer"]

    return args, pl_args


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
    args, pl_args = parse_args()
    print(args, pl_args)

    checkpoint, images_dir, out = args.checkpoint, args.images_dir, args.out
    is_regression = "regression" in str(checkpoint)
    os.makedirs(Path(out).parent, exist_ok=True)

    with torch.no_grad():
        model_constructor = LitModelRegression if is_regression else LitModelClassification
        model = model_constructor.load_from_checkpoint(str(checkpoint), batch_size=8)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        mean, std = DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD

        image_transform_train = transforms.Compose(
            [
                # transforms.RandomResizedCrop(model.image_size, scale=(0.5, 1.0)),
                transforms.Resize(model.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        predict_datamodule = GeoguesserDataModulePredict(
            [args.images_dir], num_classes=model.num_classes, batch_size=32, image_transform=image_transform_train
        )

        # use_gpu = torch.cuda.is_available()
        trainer: pl.Trainer = pl.Trainer.from_argparse_args(
            pl_args,
            log_every_n_steps=1,
            callbacks=[InferenceWriter(out)],
            checkpoint_callback=False,
            logger=False,
        )

        predictions = trainer.predict(model=model, datamodule=predict_datamodule)
        print(predictions)
