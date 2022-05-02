from __future__ import annotations, division, print_function

from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.model_summary import ModelSummary
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy

from calculate_norm_std import calculate_norm_std
from datamodule_geoguesser import GeoguesserDataModule
from defaults import DEFAULT_EARLY_STOPPING_EPOCH_FREQ, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD
from model import LitModelClassification, LitModelRegression, LitSingleModel
from model_callbacks import (
    BackboneFinetuningLastLayers,
    LogMetricsAsHyperparams,
    OnTrainEpochStartLogCallback,
    OverrideEpochMetricCallback,
)
from train_args import parse_args_train
from utils_functions import add_prefix_to_keys, get_timestamp, is_primitive, random_codeword, stdout_to_file
from utils_paths import PATH_REPORT
from utils_train import SchedulerType

if __name__ == "__main__":
    args, pl_args = parse_args_train()

    timestamp = get_timestamp()
    experiment_codeword = random_codeword()
    filename_report = Path(args.output_report, "__".join(["train", experiment_codeword, timestamp]) + ".txt")
    stdout_to_file(filename_report)
    print(str(filename_report))
    pprint([vars(args), vars(pl_args)])

    image_size = args.image_size
    num_workers = args.num_workers
    model_names = args.models
    unfreeze_blocks_num = args.unfreeze_blocks
    pretrained = args.pretrained
    learning_rate = args.lr
    trainer_checkpoint = args.trainer_checkpoint
    unfreeze_backbone_at_epoch = args.unfreeze_backbone_at_epoch
    weight_decay = args.weight_decay
    shuffle_before_splitting = args.shuffle_before_splitting
    train_frac, val_frac, test_frac = args.split_ratios
    dataset_dirs = args.dataset_dirs
    batch_size = args.batch_size
    cached_df = args.cached_df
    load_dataset_in_ram = args.load_in_ram
    use_single_images = args.use_single_images
    is_regression = args.regression
    scheduler_type = args.scheduler
    epochs = args.epochs
    recaculate_norm = args.recaculate_normalization

    mean, std = calculate_norm_std(dataset_dirs) if recaculate_norm else DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD

    image_transform_train = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    datamodule = GeoguesserDataModule(
        cached_df=cached_df,
        dataset_dirs=dataset_dirs,
        batch_size=batch_size,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        image_transform=image_transform_train,
        num_workers=num_workers,
        shuffle_before_splitting=shuffle_before_splitting,
        load_dataset_in_ram=load_dataset_in_ram,
    )
    datamodule.setup()
    num_classes = datamodule.num_classes
    experiment_directory_name = "{}__{}__{}".format(
        experiment_codeword, "regression" if is_regression else "num_classes_" + str(num_classes), timestamp
    )
    datamodule.store_df_to_report(Path(PATH_REPORT, experiment_directory_name, "data.csv"))

    log_dictionary = {
        **add_prefix_to_keys(vars(args), "user_args/"),
        **add_prefix_to_keys(vars(pl_args), "lightning_args/"),
        "train_size": len(datamodule.train_dataloader().dataset),  # type: ignore
        "val_size": len(datamodule.val_dataloader().dataset),  # type: ignore
        "test_size": len(datamodule.test_dataloader().dataset),  # type: ignore
        "num_classes": num_classes,
    }

    for model_name in model_names:
        # The EarlyStopping callback runs at the end of every validation epoch, which, under the default
        # configuration, happen after every training epoch.
        callback_early_stopping = EarlyStopping(
            monitor="val/loss_epoch",
            patience=DEFAULT_EARLY_STOPPING_EPOCH_FREQ,
            verbose=True,
            check_on_train_epoch_end=True,
        )
        callback_checkpoint = ModelCheckpoint(
            monitor="val/haversine_distance_epoch",
            filename="__".join(
                [
                    experiment_codeword,
                    "haversine_{val/haversine_distance_epoch:.4f}",
                    "val_acc_{val/acc_epoch:.2f}",
                    "val_loss_{val/loss_epoch:.2f}",
                    timestamp,
                ]
            ),
            auto_insert_metric_name=False,
        )
        bar_refresh_rate = int(len(datamodule.train_dataloader()) / pl_args.log_every_n_steps)

        callbacks = [
            callback_early_stopping,
            callback_checkpoint,
            TQDMProgressBar(refresh_rate=bar_refresh_rate),
            LogMetricsAsHyperparams(),
            OnTrainEpochStartLogCallback(),
            ModelSummary(max_depth=2),
            OverrideEpochMetricCallback(),
        ]

        if unfreeze_backbone_at_epoch and scheduler_type == SchedulerType.PLATEAU.value:
            rate_fine_tuning_multiply = 3
            learning_rate = float(learning_rate * rate_fine_tuning_multiply)
            multiplicative = lambda epoch: 1
            callbacks.append(
                BackboneFinetuningLastLayers(
                    unfreeze_blocks_num=unfreeze_blocks_num,
                    backbone_initial_ratio_lr=1 / rate_fine_tuning_multiply,
                    unfreeze_backbone_at_epoch=unfreeze_backbone_at_epoch,
                    lambda_func=multiplicative,
                ),
            )

        model_constructor = (
            LitSingleModel if use_single_images else (LitModelRegression if is_regression else LitModelClassification)
        )
        model = model_constructor(
            num_classes=num_classes,
            model_name=model_names[0],
            pretrained=pretrained,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            image_size=image_size,
            scheduler_type=scheduler_type,
            epochs=epochs,
            class_to_crs_centroid_map=datamodule.class_to_crs_centroid_map,
            crs_scaler=datamodule.crs_scaler,
            train_steps_per_epoch=len(datamodule.train_dataloader()),
        )

        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=str(PATH_REPORT),
            name=experiment_directory_name,
            default_hp_metric=False,
            log_graph=True,
        )

        tb_logger.log_hyperparams(log_dictionary)

        trainer: pl.Trainer = pl.Trainer.from_argparse_args(
            pl_args,
            logger=[tb_logger],
            default_root_dir=PATH_REPORT,
            callbacks=callbacks,
            auto_lr_find=scheduler_type == SchedulerType.AUTO_LR.value,
        )

        if scheduler_type == SchedulerType.AUTO_LR.value:
            trainer.tune(
                model, datamodule=datamodule, lr_find_kwargs={"num_training": 35, "early_stop_threshold": None}
            )

        trainer.fit(model, datamodule, ckpt_path=trainer_checkpoint)
        trainer.test(model, datamodule)
