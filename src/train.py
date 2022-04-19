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
from model import LitModel, LitSingleModel, OnTrainEpochStartLogCallback
from calculate_norm_std import calculate_norm_std
from utils_functions import get_timestamp, stdout_to_file
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
    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    cached_df = args.cached_df
    load_dataset_in_ram = args.load_in_ram
    use_single_images = args.use_single_images

    mean, std = calculate_norm_std(dataset_dir)

    image_transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    data_module = GeoguesserDataModule(
        cached_df=cached_df,
        dataset_dir=dataset_dir,
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

    for model_name in model_names:
        # The EarlyStopping callback runs at the end of every validation epoch, which, under the default
        # configuration, happen after every training epoch.
        callback_early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=DEFAULT_EARLY_STOPPING_EPOCH_FREQ,
            verbose=True,
        )
        callback_checkpoint = ModelCheckpoint(filename=model_name + "-{val_acc:.2f}-{val_loss:.2f}")
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

        model = LitModel(
            data_module=data_module,
            num_classes=data_module.train_dataset.num_classes,
            model_name=model_names[0],
            pretrained=pretrained,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            image_size=image_size,
            context_dict={**vars(args), **vars(pl_args)},
        )
        if use_single_images:
            model = LitSingleModel(
                data_module=data_module,
                num_classes=data_module.train_dataset.num_classes,
                model_name=model_names[0],
                pretrained=pretrained,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                image_size=image_size,
                context_dict={**vars(args), **vars(pl_args)},
            )

        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=str(PATH_REPORT),
            name="{}-{}".format(timestamp, model_name),
            default_hp_metric=False,
        )

        trainer: pl.Trainer = pl.Trainer.from_argparse_args(
            pl_args,
            logger=[tb_logger, tb_logger],
            default_root_dir=PATH_REPORT,
            callbacks=callbacks,
        )

        trainer.fit(model, data_module, ckpt_path=trainer_checkpoint)
        trainer.test(model, data_module)
