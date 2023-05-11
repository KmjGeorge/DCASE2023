from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import numpy as np


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""

    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.5, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0

    return warpper


def linear_rampup(rampup_length):
    """Linear rampup"""

    def warpper(epoch):
        if epoch < rampup_length:
            return epoch / rampup_length
        else:
            return 1.0

    return warpper


def linear_rampdown(rampdown_length, start=0, last_value=0):
    """Linear rampup -(start)- (rampdown_length) \ _(for the rest)  """

    def warpper(epoch):
        if epoch <= start:
            return 1.
        elif epoch - start < rampdown_length:
            return last_value + (1. - last_value) * (rampdown_length - epoch + start) / rampdown_length
        else:
            return last_value

    return warpper


def cosine_rampdown(rampdown_length, num_epochs):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""

    def warpper(epoch):
        if epoch >= (num_epochs - rampdown_length):
            ep = .5 * (epoch - (num_epochs - rampdown_length))
            return float(.5 * (np.cos(np.pi * ep / rampdown_length) + 1))
        else:
            return 1.0

    return warpper


def exp_rampdown(rampdown_length, num_epochs):
    """Exponential rampdown from https://arxiv.org/abs/1610.02242"""

    def warpper(epoch):
        if epoch >= (num_epochs - rampdown_length):
            ep = .5 * (epoch - (num_epochs - rampdown_length))
            return float(np.exp(-(ep * ep) / rampdown_length))
        else:
            return 1.0

    return warpper


def exp_warmup(rampup_length, rampdown_length, num_epochs):
    rampup = exp_rampup(rampup_length)
    rampdown = exp_rampdown(rampdown_length, num_epochs)

    def warpper(epoch):
        return rampup(epoch) * rampdown(epoch)

    return warpper


def exp_warmup_linear_down(warmup, rampdown_length, start_rampdown, last_value):
    rampup = exp_rampup(warmup)
    rampdown = linear_rampdown(rampdown_length, start_rampdown, last_value)

    def warpper(epoch):
        return rampup(epoch) * rampdown(epoch)

    return warpper


def exp_warmup_cosine_down(warmup, rampdown_length, num_epochs):
    rampup = exp_rampup(warmup)
    rampdown = cosine_rampdown(rampdown_length, num_epochs)

    def warpper(epoch):
        return rampup(epoch) * rampdown(epoch)

    return warpper


def get_scheduler_lambda(warm_up_len, ramp_down_start, ramp_down_len, last_lr_value):
    """
    @param warm_up_len: number of epochs for the lr to reach its maximum value
    @param ramp_down_start: control the epoch where decline of the lr starts
    @param ramp_down_len: number of epochs it takes for the lr to descend
    @param last_lr_value: final value of lr as a percentage of the original lr
    @param schedule_mode: method of scheduling 'exp_lin' and 'cos_cyc' are available
    @return: configured lr scheduler
    """
    return exp_warmup_linear_down(warm_up_len, ramp_down_len, ramp_down_start, last_lr_value)


def ExpWarmupLinearDownScheduler(optimizer, warm_up_len=100, ramp_down_start=250, ramp_down_len=400,
                                 last_lr_value=0.005):
    """
    @param optimizer: optimizer used for training
    @param schedule_mode: scheduling mode of the lr
    @return: updated version of the optimizer with new lr
    """
    return torch.optim.lr_scheduler.LambdaLR(optimizer,
                                             get_scheduler_lambda(warm_up_len, ramp_down_start, ramp_down_len,
                                                                  last_lr_value))


def ExpWarmupCosineDownScheduler(optimizer, warmup, rampdown_length, num_epochs):
    """
    @param optimizer: optimizer used for training
    @param schedule_mode: scheduling mode of the lr
    @return: updated version of the optimizer with new lr
    """
    return torch.optim.lr_scheduler.LambdaLR(optimizer,
                                             exp_warmup_cosine_down(warmup, rampdown_length, num_epochs))
