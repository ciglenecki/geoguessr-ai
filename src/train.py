from __future__ import annotations, division, print_function

from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import BackboneFinetuning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.model_summary import ModelSummary
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy

from calculate_norm_std import calculate_norm_std
from datamodule_geoguesser import GeoguesserDataModule
from defaults import DEFAULT_EARLY_STOPPING_EPOCH_FREQ
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

if __name__ == "__main__":
    args, pl_args = parse_args_train()

    study_name = get_timestamp() + "-" + random_codeword()
    study_name_extended = "{}{}".format(study_name, "-regression" if args.regression else "")
    filename_report = Path(args.output_report, "-".join(["train", study_name_extended]) + ".txt")
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
    auto_lr = not args.no_auto_lr

    # mean, std = calculate_norm_std(dataset_dirs)
    mean, std = [0.5006, 0.5116, 0.4869], [0.1966, 0.1951, 0.2355]

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
            filename=model_name
            + "__haversine_{val/haversine_distance_epoch:.4f}__val_acc_{val/acc_epoch:.2f}__val_loss_{val/loss_epoch:.2f}",
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

        if unfreeze_backbone_at_epoch:
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
            datamodule=datamodule,
            num_classes=num_classes,
            model_name=model_names[0],
            pretrained=pretrained,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            image_size=image_size,
        )

        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=str(PATH_REPORT),
            name="{}{}".format(study_name, "-regression" if is_regression else "-num_classes_" + str(num_classes)),
            default_hp_metric=False,
            log_graph=True,
        )

        tb_logger.log_hyperparams(log_dictionary)

        trainer: pl.Trainer = pl.Trainer.from_argparse_args(
            pl_args,
            logger=[tb_logger],
            default_root_dir=PATH_REPORT,
            callbacks=callbacks,
        )

        if auto_lr:
            lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule, num_training=5)
            if lr_finder:
                # print("Results from the lr_finder:", lr_finder.results, sep="\n")
                # lr_finder.plot(suggest=True, show=True)
                new_lr = lr_finder.suggestion()
                if new_lr:
                    print("New learning rate found by lr_finder:", new_lr)
                    model.hparams.lr = new_lr  # type: ignore
                    print(new_lr, model.learning_rate)
                    print(type(new_lr), type(model.learning_rate))

        trainer.fit(model, datamodule, ckpt_path=trainer_checkpoint)
        trainer.test(model, datamodule)
