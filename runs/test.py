#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

"""Test a video classification model for full supervised learning."""

import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pprint
import torch
import torch.nn.functional as F
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
from utils.meters import ValMeter
from models.base.builder import build_model
from datasets.base.builder import build_loader


logger = logging.get_logger(__name__)


# Collate function is now in datasets/base/builder.py


@torch.no_grad()
def test_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (data_dict, labels, indices) in enumerate(val_loader):
        if misc.get_num_gpus(cfg):
            data_dict["video"] = data_dict["video"].cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # Forward pass
        model_dict = model(data_dict)
        logits = model_dict['logits']
        loss = F.cross_entropy(logits, labels)

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(logits, labels, (1, 5))
        top1_err, top5_err = [(1.0 - x / logits.size(0)) * 100.0 for x in num_topks_correct]

        # Gather all the predictions across all the devices.
        if misc.get_num_gpus(cfg) > 1:
            loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top5_err = (loss.item(), top1_err.item(), top5_err.item())
        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(top1_err, top5_err, val_loader.batch_size * max(misc.get_num_gpus(cfg), 1))
        val_meter.update_predictions(logits, labels)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def test(cfg):
    """
    Test a video classification model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)

    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TEST.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("TEST with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    model_bucket = None
    cu.load_test_checkpoint(cfg, model, model_bucket)

    # Create the video test loader.
    val_loader = build_loader(cfg, "test")
    
    val_meter = ValMeter(len(val_loader), cfg)
    cur_epoch = 0
    test_epoch(val_loader, model, val_meter, cur_epoch, cfg)


def main():
    """
    Main entry point for testing.
    """
    from utils.config import Config
    from utils.launcher import launch_task
    
    cfg = Config(load=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
    launch_task(cfg=cfg, init_method=cfg.INIT_METHOD, func=test)
    print("Finish testing with config: {}".format(cfg.args.cfg_file))


if __name__ == "__main__":
    main()

