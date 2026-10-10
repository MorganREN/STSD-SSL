"""
STSDTrainer: Pointcept's DefaultTrainer with resumable, grouped wandb runs.
"""

from tensorboardX import SummaryWriter

import pointcept.utils.comm as comm
from pointcept.engines.train import TRAINERS, Trainer

from .tracking import init_wandb


@TRAINERS.register_module("STSDTrainer")
class STSDTrainer(Trainer):
    def build_writer(self):
        if not comm.is_main_process():
            return None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb:
            init_wandb(self.cfg)
        return SummaryWriter(self.cfg.save_path)
