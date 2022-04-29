from pytorch_lightning.callbacks import Callback


class OverrideEpochMetricCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def training_epoch_end(self, trainer, pl_module):
        self._log_step_as_current_epoch(trainer, pl_module)

    def test_epoch_end(self, trainer, pl_module):
        self._log_step_as_current_epoch(trainer, pl_module)

    def validation_epoch_end(self, trainer, pl_module):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer, pl_module):
        pl_module.log("step", trainer.current_epoch)
