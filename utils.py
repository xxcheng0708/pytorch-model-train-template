from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
import math
import torch


class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class CosineAnnealingLRWarmup(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=1.0e-5, last_epoch=-1, verbose=False,
                 warmup_steps=2, warmup_start_lr=1.0e-5):
        super(CosineAnnealingLRWarmup, self).__init__(optimizer, T_max=T_max,
                                                      eta_min=eta_min,
                                                      last_epoch=last_epoch)
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        if warmup_steps > 0:
            self.base_warup_factors = [
                (base_lr / warmup_start_lr) ** (1.0 / self.warmup_steps)
                for base_lr in self.base_lrs
            ]

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if hasattr(self, 'warmup_steps'):
            if self.last_epoch < self.warmup_steps:
                return [self.warmup_start_lr * (warmup_factor ** self.last_epoch)
                        for warmup_factor in self.base_warup_factors]
            else:
                return [self.eta_min + (base_lr - self.eta_min) *
                        (1 + math.cos(
                            math.pi * (self.last_epoch - self.warmup_steps) / (self.T_max - self.warmup_steps))) * 0.5
                        for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                    for base_lr in self.base_lrs]


def evaluate_accuracy_and_loss(data_iter, model, loss, accelerator=None, is_half=False, local_rank=-1, world_size=1):
    acc_sum = 0.0
    loss_sum = 0.0
    n = 0

    with torch.no_grad():
        for X, y in data_iter:
            X = X.cuda()
            if is_half:
                X = X.half()
            y = y.cuda()
            y_pred = model(X)

            # if local_rank == 0:
            #     print("evaluate, local rank: {}, {}, {}, {}".format(local_rank, X.shape, y.shape, y_pred.shape))

            # 适用于非分布式场景
            y_gather = y.clone().detach()
            y_pred_gather = y_pred.clone().detach()

            # 适用于accelerate
            if accelerator:
                # X = accelerator.gather(X)
                y_gather = accelerator.gather(y)
                y_pred_gather = accelerator.gather(y_pred)
            elif local_rank != -1:  # 适用于DDP、FSDP
                # torch.distributed.all_gather_into_tensor(X, X)
                y_gather = torch.zeros_like(y).repeat(world_size)
                y_pred_gather = torch.zeros_like(y_pred).repeat((world_size, 1))
                torch.distributed.all_gather_into_tensor(y_gather, y)
                torch.distributed.all_gather_into_tensor(y_pred_gather, y_pred)
                # print("y_gather: {}, y_pred_gather: {}".format(y_gather.shape, y_pred_gather.shape))

            # if local_rank == 0:
            #     print("evaluate, local rank: {}, {}, {}, {}".format(local_rank, X.shape, y_gather.shape, y_pred_gather.shape))
            acc_sum += (y_pred_gather.argmax(dim=1) == y_gather).sum().item()
            loss_sum += loss(y_pred_gather, y_gather).sum().item()
            n += y_gather.shape[0]

            # break
        return acc_sum / n, loss_sum / n
