from typing import Any, Dict, Literal

from pytorch_lightning.utilities.types import _LRScheduler as _pl_LRScheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class DelayerScheduler(_LRScheduler):
    """Starts with a flat lr schedule until it reaches N epochs the applies a scheduler
    Args:
            optimizer (Optimizer): Wrapped optimizer.
            delay_epochs: number of epochs to keep the initial lr until starting aplying the scheduler
            after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    optimizer: Any
    last_epoch: int

    def __init__(
        self,
        delay_epochs: int,
        after_scheduler: _LRScheduler,
        optimizer: Optimizer,
        train_dataloader_size: int,
        interval_mode: Literal["epoch", "step"],
        warmup_lr: float,
    ):
        self.delay_epochs_ = delay_epochs
        self.after_scheduler = after_scheduler
        # self.train_dataloader_size_ = train_dataloader_size
        self.step_stop_ = delay_epochs * train_dataloader_size
        self.interval_mode_ = interval_mode
        self.warmup_lr_ = warmup_lr
        self.finished = False
        super().__init__(optimizer)

    def _get_after_scheduler_lr(self):
        if hasattr(self.after_scheduler, "get_lr"):
            return self.after_scheduler.get_lr()
        return self._last_lr  # type: ignore

    def get_lr(self):
        if self.finished:
            return self._get_after_scheduler_lr()
        # self.last_epoch  = self.optimizer._step_count = STEP in case of cycle
        is_finished_epoch = self.interval_mode_ == "epoch" and self.last_epoch >= self.delay_epochs_
        is_finished_step = self.interval_mode_ == "step" and self.optimizer._step_count >= self.step_stop_
        print("is_finished_epoch", is_finished_epoch)
        print("is_finished_step", is_finished_step)
        if is_finished_epoch or is_finished_step:
            self.after_scheduler.base_lrs = self.base_lrs  # type: ignore
            self.finished = True
            return self._get_after_scheduler_lr()

        return [self.warmup_lr_]

    def step(self, *args, **kwars):
        print(args, kwars)
        print("Delay_epoch", self.delay_epochs_)
        print("Last_epoch", self.last_epoch)
        print("Optim step count", self.optimizer._step_count)
        print("step_stop_", self.step_stop_)
        print("Finished", self.finished)
        if self.finished:
            self.after_scheduler.step(*args, **kwars)
        else:
            return super().step()

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


# class DelayerSchedulerOrg(_LRScheduler):
#     """Starts with a flat lr schedule until it reaches N epochs the applies a scheduler
#     Args:
#             optimizer (Optimizer): Wrapped optimizer.
#             delay_epochs: number of epochs to keep the initial lr until starting aplying the scheduler
#             after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
#     """

#     def __init__(self, optimizer, delay_epochs, after_scheduler: _LRScheduler, warmup_lr: int):
#         self.delay_epochs = delay_epochs
#         self.after_scheduler = after_scheduler
#         self.finished = False
#         self.warmup_lr = warmup_lr
#         super().__init__(optimizer)

#     def state_dict(self):
#         """Returns the state of the scheduler as a :class:`dict`.

#         It contains an entry for every variable in self.__dict__ which
#         is not the optimizer.
#         """
#         state_dict = {key: value for key, value in self.__dict__.items() if key != "optimizer"}
#         state_dict["after_scheduler_dict"] = self.after_scheduler.state_dict()
#         return state_dict

#     def load_state_dict(self, state_dict):
#         """Loads the schedulers state.

#         Args:
#             state_dict (dict): scheduler state. Should be an object returned
#                 from a call to :meth:`state_dict`.
#         """
#         self.__dict__.update(state_dict)
#         self.after_scheduler.__dict__.update(state_dict["after_scheduler_dict"])

#     def get_lr(self):
#         if self.last_epoch >= self.delay_epochs:  # type: ignore
#             if not self.finished:
#                 self.after_scheduler.base_lrs = self.base_lrs  # type: ignore
#                 self.finished = True
#             return self.after_scheduler.get_lr()
#         return [self.warmup_lr]

#     def step(self, epoch=None):
#         if self.finished:
#             if epoch is None:
#                 self.after_scheduler.step(None)
#             else:
#                 self.after_scheduler.step(epoch - self.delay_epochs)
#         else:
#             return super().step(epoch)
