from typing import Any, Dict
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning.utilities.types import _LRScheduler as _pl_LRScheduler


class DelayerScheduler(_LRScheduler):
    """Starts with a flat lr schedule until it reaches N epochs the applies a scheduler
    Args:
            optimizer (Optimizer): Wrapped optimizer.
            delay_epochs: number of epochs to keep the initial lr until starting aplying the scheduler
            after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    _step_count: int
    optimizer: Any
    last_epoch: int

    def __init__(self, delay_epochs, after_scheduler, optimizer, train_dataloader_size):
        self.delay_epochs = delay_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        self.train_dataloader_size = train_dataloader_size
        self.step_stop = delay_epochs * train_dataloader_size
        super().__init__(optimizer)

    def get_lr(self):
        optimizer_step = self.optimizer._step_count
        if optimizer_step >= self.step_stop:  # type: ignore
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs  # type: ignore
                self.finished = True
            return self.after_scheduler.get_lr()

        return self.base_lrs  # type: ignore

    def step(self, epoch=None):
        print(self.optimizer._step_count, self.step_stop, self.finished)
        if self.finished:
            self.after_scheduler.step(None)
        else:
            return super().step(epoch)


class DelayerSchedulerOrg(_LRScheduler):
    """Starts with a flat lr schedule until it reaches N epochs the applies a scheduler
    Args:
            optimizer (Optimizer): Wrapped optimizer.
            delay_epochs: number of epochs to keep the initial lr until starting aplying the scheduler
            after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, delay_epochs, after_scheduler: _LRScheduler, warmup_lr: int):
        self.delay_epochs = delay_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        self.warmup_lr = warmup_lr
        super().__init__(optimizer)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key != "optimizer"}
        state_dict["after_scheduler_dict"] = self.after_scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
        self.after_scheduler.__dict__.update(state_dict["after_scheduler_dict"])

    def get_lr(self):
        if self.last_epoch >= self.delay_epochs:  # type: ignore
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs  # type: ignore
                self.finished = True
            return self.after_scheduler.get_lr()
        return [self.warmup_lr]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.delay_epochs)
        else:
            return super().step(epoch)
