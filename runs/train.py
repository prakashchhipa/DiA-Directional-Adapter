#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

"""Train a video classification model for full supervised learning."""

import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import numpy as np
import pprint
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import models.utils.optimizer as optim
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
from utils.meters import TrainMeter, ValMeter
from models.base.builder import build_model
from datasets.base.builder import build_loader, shuffle_dataset

logger = logging.get_logger(__name__)


# Collate function is now in datasets/base/builder.py


def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, val_meter, val_loader):
    # Enable train mode.
    model.train()
    norm_train = False
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm3d, nn.LayerNorm)) and module.training:
            norm_train = True
    logger.info(f"Norm training: {norm_train}")
    train_meter.iter_tic()

    for cur_iter, (data_dict, labels, indices) in enumerate(train_loader):
        # Move to GPU
        if misc.get_num_gpus(cfg):
            data_dict["video"] = data_dict["video"].cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch, cfg)
        optim.set_lr(optimizer, lr)

        # Forward pass
        model_dict = model(data_dict)
        logits = model_dict['logits']

        # Compute loss with optional label smoothing
        label_smoothing = getattr(cfg.AUGMENTATION, 'LABEL_SMOOTHING', 0.0)
        if label_smoothing > 0.0:
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        else:
            loss = F.cross_entropy(logits, labels)

        # check Nan Loss.
        if math.isnan(loss):
            loss.backward(retain_graph=False)
            optimizer.zero_grad()
            continue
        
        loss.backward(retain_graph=False)

        # Optimize
        optimizer.step()
        optimizer.zero_grad()

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(logits, labels, (1, 5))
        top1_err, top5_err = [(1.0 - x / logits.size(0)) * 100.0 for x in num_topks_correct]

        # Gather all the predictions across all the devices.
        if misc.get_num_gpus(cfg) > 1:
            loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top5_err = (loss.item(), top1_err.item(), top5_err.item())

        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats(top1_err, top5_err, loss, lr, train_loader.batch_size * max(misc.get_num_gpus(cfg), 1))
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (data_dict, labels, indices) in enumerate(val_loader):
        if misc.get_num_gpus(cfg):
            data_dict["video"] = data_dict["video"].cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # Forward pass
        model_dict = model(data_dict)
        logits = model_dict['logits']
        
        # Compute loss (label smoothing typically disabled for validation)
        # Note: Label smoothing is usually only applied during training for regularization
        # During validation, we evaluate on true labels, so label_smoothing=0.0 is recommended
        label_smoothing = getattr(cfg.AUGMENTATION, 'LABEL_SMOOTHING', 0.0)
        # For validation, we typically don't use label smoothing (set to 0.0 in config)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.0)

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


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
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
    logging.setup_logging(cfg, cfg.TRAIN.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    model_bucket = None

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, model_bucket)

    # Create the video train and val loaders.
    train_loader = build_loader(cfg, "train")
    val_loader = build_loader(cfg, "test") if cfg.TRAIN.EVAL_PERIOD != 0 else None

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg) if val_loader is not None else None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    # Freeze parameters - only adapter is trainable
    for name, param in model.named_parameters():
        if 'Adapter' not in name and 'classification_layer' not in name:
            param.requires_grad = False

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info('Trainable: {}'.format(name))

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

    # Training loop
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        shuffle_dataset(train_loader, cur_epoch)
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, val_meter, val_loader)
        
        # Save checkpoint
        if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, model_bucket)
        
        # Evaluate
        if val_loader is not None and (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)
            model.train()


def main():
    """
    Main entry point for training.
    """
    from utils.config import Config
    from utils.launcher import launch_task
    
    cfg = Config(load=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
    launch_task(cfg=cfg, init_method=cfg.INIT_METHOD, func=train)
    print("Finish training with config: {}".format(cfg.args.cfg_file))


if __name__ == "__main__":
    main()

