#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

import torch.nn as nn
from models.base.adapter import HEAD_REGISTRY


class BaseVideoModel(nn.Module):
    def __init__(self, cfg):
        super(BaseVideoModel, self).__init__()
        self.cfg = cfg
        self.model = HEAD_REGISTRY.get(cfg.ADAPTER.NAME)(cfg=cfg)
        self.task_type = getattr(cfg, "TASK_TYPE", "few_shot_action")

    def forward(self, x):
        # Check if this is standard classification (video + labels format)
        if self.task_type in ["classification", "full_supervised"]:
            if isinstance(x, dict) and "video" in x:
                # Standard classification format: {"video": tensor, "labels": tensor}
                video = x["video"]
                return self.model.forward_classification(video, x.get("labels", None))
        # Default: few-shot format
        return self.model(x)

    def train(self, mode=True):
        self.training = mode
        super(BaseVideoModel, self).train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)

        return self
