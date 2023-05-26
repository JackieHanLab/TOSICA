import numpy as np
import torch
import random
import pickle
import torch.optim as optim
from .log_util import logger


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).
    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception as identifier:
        pass


def load_train_dictionary(training_dictionary_file):
    with open(training_dictionary_file,'rb') as fd:
        training_dictionary = pickle.load(fd)
        return training_dictionary


def load_best_model_config(training_dictionary):
    min_nll_test_active = float("inf")
    for epoch, training_values in training_dictionary.items():
        nll_test_active = training_values[0]
        if nll_test_active < min_nll_test_active:
            best_epoch = epoch
            min_nll_test_active = nll_test_active
    return best_epoch, min_nll_test_active


def get_device(device_id=0):
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > device_id:
            device = f"cuda:{device_id}"
        else:
            device = "cuda:0"
    else:
        device = "cpu"
    logger.info(f'device: {device}')
    return device


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if self.warmup > 0  and epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor