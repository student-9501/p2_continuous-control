"""
a wrapper for weights and biases so we can run training sessions without having an account configured
"""
import os

import wandb


class Wand:
    """ so the trainer can run without a wandb account """
    def __init__(self, agent):
        self.agent = agent

    def login(self):
        if not self.configured:
            return
        wandb.login()

    def init(self, **kwargs):
        if not self.configured:
            return
        print('saving wandb', kwargs)
        res = wandb.init(**kwargs)
        print(res)

    def log(self, arg):
        if not self.configured:
            return
        wandb.log(arg)

    @property
    def configured(self):
        return os.environ.get('WANDB_ENTITY')
