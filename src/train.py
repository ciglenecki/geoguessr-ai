from __future__ import annotations, division, print_function
import os

from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.model_summary import ModelSummary
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from torchvision import transforms

from calculate_norm_std import calculate_norm_std
from callback_backbone_last_layers import BackboneFinetuningLastLayers
from datamodule_geoguesser import GeoguesserDataModule
from config import DEFAULT_EARLY_STOPPING_EPOCH_FREQ, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD
from model_callbacks import (
    BackboneFinetuning,
    BackboneFreezing,
    LogMetricsAsHyperparams,
    OnTrainEpochStartLogCallback,
    OverrideEpochMetricCallback,
)
from model_classification import LitModelClassification, LitSingleModel
from model_regression import LitModelRegression
from utils_dataset import DatasetSplitType, get_dataset_dirs_uuid_paths
from train_args import parse_args_train
from utils_functions import add_prefix_to_keys, get_timestamp, random_codeword, stdout_to_file
from utils_paths import PATH_REPORT_QUICK
from utils_train import SchedulerType

if __name__ == "__main__":
    args, pl_args = parse_args_train()
    image_size = args.image_size
    num_workers = args.num_workers
    model_name = args.model
    unfreeze_blocks_num = args.unfreeze_blocks
    pretrained = args.pretrained
    learning_rate = args.lr
    trainer_checkpoint = args.trainer_checkpoint
    unfreeze_at_epoch = args.unfreeze_at_epoch
    weight_decay = args.weight_decay
    shuffle_before_splitting = args.shuffle_before_splitting
    dataset_frac = args.dataset_frac
    dataset_dirs = args.dataset_dirs
    batch_size = args.batch_size
    csv_rich_static = args.csv_rich_static
    use_single_images = args.use_single_images
    is_regression = args.regression
    scheduler_type = args.scheduler
    epochs = args.epochs
    recaculate_norm = args.recaculate_normalization
    optimizer_type = args.optimizer
    is_quick = args.quick or dataset_frac < 0.1
    output_report = PATH_REPORT_QUICK if is_quick else args.output_report
    lr_finetune = args.lr_finetune

    timestamp = get_timestamp()
    experiment_codeword = random_codeword()

    os.makedirs(output_report, exist_ok=True)
    filename_report = Path(
        output_report,
        "__".join(["train", experiment_codeword, timestamp]) + ".txt",
    )

    stdout_to_file(filename_report)
    print(str(filename_report))
    pprint([vars(args), vars(pl_args)])

    mean, std = DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD

    datamodule = GeoguesserDataModule(
        csv_rich_static=csv_rich_static,
        dataset_dirs=dataset_dirs,
        batch_size=batch_size,
        dataset_frac=dataset_frac,
        image_size=image_size,
        num_workers=num_workers,
        shuffle_before_splitting=shuffle_before_splitting,
        train_mean_std=(mean, std),
    )
    datamodule.setup()
    num_classes = datamodule.num_classes
    experiment_directory_name = "{}__{}__{}".format(
        experiment_codeword, "regression" if is_regression else "num_classes_" + str(num_classes), timestamp
    )
    datamodule.store_df_to_report(Path(output_report, experiment_directory_name, "data_runtime.csv"))

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
                timestamp,
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
                timestamp,
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
        LitSingleModel if use_single_images else (LitModelRegression if is_regression else LitModelClassification)
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
        lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule, num_training=100)

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
