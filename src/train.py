from __future__ import annotations, division, print_function

from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy

from args_train import parse_args_train
from callback_finetuning_last_n_layers import BackboneFinetuningLastLayers
from data_module_geoguesser import GeoguesserDataModule
from defaults import DEFAULT_EARLY_STOPPING_EPOCH_FREQ
from model import LitModel, LitSingleModel, OnTrainEpochStartLogCallback, LitModelReg
from utils_functions import add_prefix_to_keys, get_timestamp, is_primitive, stdout_to_file
from calculate_norm_std import calculate_norm_std
from utils_paths import PATH_REPORT

if __name__ == "__main__":
    args, pl_args = parse_args_train()

    timestamp = get_timestamp()
    filename_report = Path(args.output_report, "-".join(["train", *args.models, timestamp]) + ".txt")
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

    mean, std = calculate_norm_std(dataset_dirs)
    # mean, std = [0.5006, 0.5116, 0.4869], [0.1966, 0.1951, 0.2355]

    image_transform_train = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    data_module = GeoguesserDataModule(
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
    data_module.setup()

    log_dictionary = {
        **add_prefix_to_keys(vars(args), "user_args/"),
        **add_prefix_to_keys(vars(pl_args), "lightning_args/"),
        "train_size": len(data_module.train_dataloader()),
        "val_size": len(data_module.val_dataloader()),
        "test_size": len(data_module.test_dataloader()),
    }

    for model_name in model_names:
        # The EarlyStopping callback runs at the end of every validation epoch, which, under the default
        # configuration, happen after every training epoch.
        callback_early_stopping = EarlyStopping(
            monitor="val/loss",
            patience=DEFAULT_EARLY_STOPPING_EPOCH_FREQ,
            verbose=True,
        )
        callback_checkpoint = ModelCheckpoint(
            monitor="val/haversine_distance",
            filename=model_name + "__haversine_{val/haversine_distance:.4f}__val_acc_{val/acc:.2f}__val_loss_{val/loss:.2f}",
            auto_insert_metric_name=False,
        )
        bar_refresh_rate = int(len(data_module.train_dataloader()) / pl_args.log_every_n_steps)

        callbacks = [
            callback_early_stopping,
            TQDMProgressBar(refresh_rate=bar_refresh_rate),
            callback_checkpoint,
            OnTrainEpochStartLogCallback(),
        ]

        if unfreeze_backbone_at_epoch:
            multiplicative = lambda epoch: 1.4
            callbacks.append(
                BackboneFinetuningLastLayers(
                    unfreeze_blocks_num=unfreeze_blocks_num,
                    unfreeze_backbone_at_epoch=unfreeze_backbone_at_epoch,
                    lambda_func=multiplicative,
                )
            )

        model_constructor = LitSingleModel if use_single_images else LitModelReg
        model = model_constructor(
            data_module=data_module,
            num_classes=data_module.train_dataset.num_classes,
            model_name=model_names[0],
            pretrained=pretrained,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            image_size=image_size,
        )

        # Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is called without a metric (otherwise calls to log_hyperparams without a metric are ignored).
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=str(PATH_REPORT),
            name="{}-{}".format(timestamp, model_name),
            default_hp_metric=True,
            log_graph=True,
        )

        tb_logger.log_hyperparams(log_dictionary)

        trainer: pl.Trainer = pl.Trainer.from_argparse_args(
            pl_args,
            logger=[tb_logger],
            default_root_dir=PATH_REPORT,
            callbacks=callbacks,
        )

        trainer.fit(model, data_module, ckpt_path=trainer_checkpoint)
        trainer.test(model, data_module)
