from __future__ import annotations, division, print_function

import inspect
import os
import time
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import List

import pytorch_lightning as pl
import yaml
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    conint,
    create_model,
    create_model_from_namedtuple,
    validator,
)
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.model_summary import ModelSummary
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.utilities.argparse import get_init_arguments_and_types

from callback_backbone_last_layers import BackboneFinetuningLastLayers
from config import DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD, cfg
from datamodule_geoguesser import GeoguesserDataModule
from logger import log, log_add_stream_handlers
from model_callbacks import (
    LogMetricsAsHyperparams,
    OnTrainEpochStartLogCallback,
    OverrideEpochMetricCallback,
)
from model_classification import LitModelClassification, LitSingleModel
from model_regression import LitModelRegression
from utils_functions import add_prefix_to_keys, generate_name, stdout_to_file
from utils_train import SchedulerType


class Experiment:
    def __init__(self):
        self.current_timestamp = time.time()
        self.codeword = generate_name(self.current_timestamp)
        self.datetime_curr = datetime.fromtimestamp(self.current_timestamp).strftime(
            cfg.datetime_format
        )
        self.experiment_id = f"{self.datetime_curr}_{self.codeword}"
        self.report_filepath = None

    def build_report_file(
        self,
        report_dir: str | Path,
        prefix_tags: list[str] = [],
        suffix_tags: list[str] = [],
        delimiter="__",
        extension="txt",
    ):
        os.makedirs(report_dir, exist_ok=True)
        filename = f"{delimiter.join([*prefix_tags, self.experiment_id, *suffix_tags])}.{extension}"
        self.report_filepath = Path(report_dir, filename)
        return open(self.report_filepath, "w+")


def main():

    # TODO: add strict config validation

    experiment = Experiment()
    # report_file = experiment.build_report_file(cfg.paths.reports)
    # log_add_stream_handlers(log, [report_file])
    log.info("Log file: %s", experiment.report_filepath)
    log.info("Config\n%s", yaml.dump(cfg.dict()))

    mean, std = DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD

    datamodule = GeoguesserDataModule(
        dataset_csv=cfg.datamodule.dataset_csv,
        dataset_dirs=cfg.datamodule.dataset_dirs,
        image_size=cfg.datamodule.image_size,
        batch_size=cfg.datamodule.batch_size,
        dataset_frac=cfg.datamodule.dataset_frac,
        num_workers=cfg.datamodule.num_workers,
        drop_last=cfg.datamodule.drop_last,
        shuffle_before_splitting=cfg.datamodule.shuffle_before_splitting,
        use_single_images=cfg.datamodule.use_single_images,
        train_mean_std=(mean, std),
    )
    datamodule.setup()
    num_classes = datamodule.num_classes
    experiment_directory_name = "{}__{}__{}".format(
        experiment_codeword,
        "regression" if is_regression else "num_classes_" + str(num_classes),
        datetime_curr,
    )
    datamodule.store_df_to_report(
        Path(output_report, experiment_directory_name, "data_runtime.csv")
    )

    train_dataloader_size = len(datamodule.train_dataloader())
    log_dictionary = {
        **add_prefix_to_keys(vars(args), "user_args/"),
        **add_prefix_to_keys(vars(pl_args), "lightning_args/"),
        "train_size": len(datamodule.train_dataloader().dataset),  # type: ignore
        "val_size": len(datamodule.val_dataloader().dataset),  # type: ignore
        "test_size": len(datamodule.test_dataloader().dataset),  # type: ignore
        "num_classes": num_classes,
    }

    # The EarlyStopping callback runs at the end of every validation epoch, which, under the default
    # configuration, happen after every training epoch.

    # Early stopping doesnt worth with --val_intreval_check

    # callback_early_stopping = EarlyStopping(
    #     monitor="val/haversine_distance_epoch",
    #     mode="min",
    #     patience=DEFAULT_EARLY_STOPPING_EPOCH_FREQ,
    #     check_on_train_epoch_end=False,  # note: this is extremely important for model checkpoint loading
    #     verbose=True,
    # )

    callback_checkpoint = ModelCheckpoint(
        monitor="val/haversine_distance_epoch",
        mode="min",
        filename="__".join(
            [
                experiment_codeword,
                "haversine_{val/haversine_distance_epoch:.4f}",
                "val_acc_{val/acc_epoch:.4f}",
                "val_loss_{val/loss_epoch:.4f}",
                datetime_curr,
            ]
        ),
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,  # note: this is extremely important for model checkpoint loading
        verbose=True,
    )

    callback_checkpoint_val = ModelCheckpoint(
        monitor="val/loss_epoch",
        mode="min",
        filename="__".join(
            [
                experiment_codeword,
                "haversine_{val/haversine_distance_epoch:.4f}",
                "val_acc_{val/acc_epoch:.4f}",
                "val_loss_{val/loss_epoch:.4f}",
                "val",
                datetime_curr,
            ]
        ),
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,  # note: this is extremely important for model checkpoint loading
        verbose=True,
    )

    bar_refresh_rate = int(train_dataloader_size / pl_args.log_every_n_steps)

    callbacks = [
        callback_checkpoint,
        callback_checkpoint_val,
        # callback_early_stopping,
        TQDMProgressBar(refresh_rate=bar_refresh_rate),
        ModelSummary(max_depth=3),
        LogMetricsAsHyperparams(),
        OverrideEpochMetricCallback(),
        OnTrainEpochStartLogCallback(),
        LearningRateMonitor(log_momentum=True),
    ]

    if unfreeze_at_epoch:
        callbacks.append(
            BackboneFinetuningLastLayers(
                unfreeze_blocks_num=unfreeze_blocks_num,
                unfreeze_at_epoch=unfreeze_at_epoch,
                lr_finetuning_range=[learning_rate, learning_rate],
                lr_after_finetune=learning_rate,
            ),
        )

    model_constructor = (
        LitSingleModel
        if use_single_images
        else (LitModelRegression if is_regression else LitModelClassification)
    )

    model = model_constructor(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        learning_rate=learning_rate,
        lr_finetune=lr_finetune,
        weight_decay=weight_decay,
        batch_size=batch_size,
        image_size=image_size,
        scheduler_type=scheduler_type,
        epochs=epochs,
        class_to_crs_centroid_map=datamodule.class_to_crs_centroid_map,
        class_to_crs_weighted_map=datamodule.class_to_crs_weighted_map,
        crs_scaler=datamodule.crs_scaler,
        train_dataloader_size=train_dataloader_size,
        optimizer_type=optimizer_type,
        unfreeze_at_epoch=unfreeze_at_epoch,
    )

    tensorboard_logger = pl_loggers.TensorBoardLogger(
        save_dir=str(output_report),
        name=experiment_directory_name,
        default_hp_metric=False,  # default_hp_metric should be turned off unless you log hyperparameters (logger.log_hyperparams(dict)) before the module starts with training
        log_graph=True,
    )
    tensorboard_logger.log_hyperparams(log_dictionary)

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        pl_args,
        logger=[tensorboard_logger],
        default_root_dir=output_report,
        callbacks=callbacks,
        auto_lr_find=scheduler_type == SchedulerType.AUTO_LR.value,
    )

    if scheduler_type == SchedulerType.AUTO_LR.value:
        lr_finder = trainer.tuner.lr_find(
            model, datamodule=datamodule, num_training=100
        )

        # Results can be found in
        lr_finder.results

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.savefig("best_auti_lr.png")
        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        print(new_lr)
        exit(1)

    trainer.fit(model, datamodule, ckpt_path=trainer_checkpoint)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
