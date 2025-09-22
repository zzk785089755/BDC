# Copyright (c) OpenMMLab. All rights reserved.
# modified from megvii-bevdepth.
import os
import glob

import torch
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook

from .ema import MEGVIIEMAHook

@HOOKS.register_module()
class MEGVIIEMAHook2(MEGVIIEMAHook):
    """EMAHook used in BEVDepth.

    Modified from https://github.com/Megvii-Base
    Detection/BEVDepth/blob/main/callbacks/ema.py.
    """

    def __init__(self, *args, max_keep_ckpts=5, **kwargs):
        super(MEGVIIEMAHook2, self).__init__(*args, **kwargs)       
        self.max_keep_ckpts = max_keep_ckpts
    
    
    def after_train_epoch(self, runner):
        self.save_checkpoint(runner)

    @master_only
    def save_checkpoint(self, runner):
        state_dict = runner.ema_model.ema.state_dict()
        ema_checkpoint = {
            'epoch': runner.epoch,
            'state_dict': state_dict,
            'updates': runner.ema_model.updates
        }
        save_path = f'epoch_{runner.epoch+1}_ema.pth'
        save_path = os.path.join(runner.work_dir, save_path)
        torch.save(ema_checkpoint, save_path)
        runner.logger.info(f'Saving ema checkpoint at {save_path}')

        # delete old checkpoints     
        pattern = "epoch_*_ema.pth"  
        files = glob.glob(os.path.join(runner.work_dir, pattern))
        files.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
        if len(files) > self.max_keep_ckpts:
            files = files[:-self.max_keep_ckpts]
            for file in files:
                runner.logger.info(f'Remove ema checkpoint: {file}')
                os.remove(file)