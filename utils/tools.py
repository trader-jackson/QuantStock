import numpy as np
import torch
import shortuuid
from types import SimpleNamespace


import math

def adjust_learning_rate(optimizer, epoch, args):
    """
    args.lradj options
    ------------------
    * 'type1'  – step decay: LR × 0.5 every args.adjust_interval epochs
    * 'type2'  – hard‑coded table
    * 'cosine' – CosineAnnealing from LR_max → LR_min over args.train_epochs
                optional linear warm‑up for the first args.warmup epochs
    """

    lr = None                                 # will hold the new learning‑rate

    # --- type‑1: step decay -------------------------------------------------
    if args.lradj == 'type1':
        if epoch % args.adjust_interval == 0:
            k      = (epoch // args.adjust_interval) - 1
            lr     = args.learning_rate * (0.5 ** max(k, 0))

    # --- type‑2: fixed table -----------------------------------------------
    elif args.lradj == 'type2':
        table = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        lr = table.get(epoch, None)

    # --- NEW: cosine schedule ----------------------------------------------
    elif args.lradj == 'cosine':
        lr_max   = args.learning_rate           # initial LR
        lr_min   = getattr(args, "lr_min", 0)   # default 0 – can be set in args
        T        = args.train_epochs
        warmup   = getattr(args, "warmup_epochs", 0)

        if epoch < warmup:                      # linear warm‑up
            lr = lr_max * (epoch + 1) / warmup
        else:
            t     = epoch - warmup
            T_cos = T  - warmup
            cos_inner = math.pi * t / T_cos
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(cos_inner))

    # -----------------------------------------------------------------------

    # If we decided on a new LR, push it into the optimiser
    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"Updating learning rate to {lr:.6g}")


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def generate_id():
    """Identical to wandb.util.generate_id()
    A drop-in replacement for environments where wandb is unavailable
    """
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return run_gen.random(8)



def dict_to_namespace(d):
    """Recursively convert a dictionary to a SimpleNamespace"""
    for key, value in d.items():
        if isinstance(value, dict):
            pass
    return SimpleNamespace(**d)