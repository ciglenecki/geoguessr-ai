class LitModelClassification(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_name: str,
        pretrained: bool,
        image_size: int,
    ):
        super().__init__()

        self.image_size = image_size

        backbone = torch.hub.load(
            DEFAULT_TORCHVISION_VERSION, model_name, pretrained=pretrained
        )
        self.backbone = model_remove_fc(backbone)
        self.fc = nn.Linear(
            get_last_fc_in_channels(self.image_size, self.batch_size), num_classes
        )

        self.save_hyperparameters()

    def forward(self, image_list):
        out = self.backbone(image_list)
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        image_list, y, _image_list, y = batch
        y_pred = self(image_list)
        loss = F.cross_entropy(y_pred, y)
        acc = caculate_accuracy(y_pred, y)
        data_dict = {"loss": loss, "train/loss": loss, "train/acc": acc}
        return data_dict

    def validation_step(self, batch, batch_idx):
        image_list, y_true = batch
        y_pred = self(image_list)
        loss = F.cross_entropy(y_pred, y_true)
        acc = caculate_accuracy(y_pred, y_true)
        data_dict = {"loss": loss, "val/loss": loss, "val/acc": acc}
        return data_dict

    def test_step(self, batch, batch_idx):
        image_list, y_true = batch
        y_pred = self(image_list)
        loss = F.cross_entropy(y_pred, y_true)
        acc = caculate_accuracy(y_pred, y_true)
        data_dict = {"loss": loss, "test/loss": loss, "test/acc": acc}
        return data_dict

    def configure_optimizers(self):
        wait_n_epochs_before_reducing_learning_rate = 5
        reduce_learning_rate_by_factor_of = 0.1  # from 10 to 1, from 0.01 to 0.001...
        learning_rate = 1e-4

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=reduce_learning_rate_by_factor_of,
            patience=wait_n_epochs_before_reducing_learning_rate,
        )

        config_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # How many epochs/steps should pass between calls to `scheduler.step()`?
                # 1: check if learning rate has to be updated after every epoch/step
                "frequency": 1,
                "interval": "epoch",
                # what metric to monitor?
                # Reduce learning rate when val loss isn't going down anymore
                "monitor": "val/loss_epoch",
            },
        }

        return config_dict
