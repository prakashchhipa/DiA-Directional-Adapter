#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

""" Builder for the dataloader."""

import torch
import utils.misc as misc
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from datasets.base.few_shot_dataset import Few_shot
from datasets.base.classification_dataset import Classification


def classification_collate_fn(batch):
    """
    Custom collate function for standard classification.
    Converts dataset output format to model input format.
    """
    videos = []
    labels = []
    indices = []
    
    for item in batch:
        data, label_dict, index, meta = item
        videos.append(data["video"])  # [C, T, H, W]
        labels.append(label_dict["supervised"])
        indices.append(index)
    
    # Stack videos: [B, C, T, H, W] -> [B*T, C, H, W]
    videos = torch.stack(videos, dim=0)  # [B, C, T, H, W]
    B, C, T, H, W = videos.shape
    videos = videos.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
    videos = videos.view(B * T, C, H, W)  # [B*T, C, H, W]
    
    # Stack labels
    labels = torch.tensor(labels, dtype=torch.long)
    
    return {"video": videos, "labels": labels}, labels, indices


def get_sampler(cfg, dataset, split, shuffle):
    if misc.get_num_gpus(cfg) > 1:
        return DistributedSampler(dataset, shuffle=shuffle)
    else:
        return None


def build_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (Configs): global config object. details in utils/config.py
        split (str): the split of the data loader. Options include `train`,
            `val`, `test`, and `submission`.
    Returns:
        loader object.
    """
    assert split in ["train", "val", "test", "submission"]
    if split in ["train"]:
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    else:
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(cfg, split)

    # Create a sampler for multi-process training
    sampler = get_sampler(cfg, dataset, split, shuffle)
    
    # Set collate function for standard classification
    collate_fn = None
    task_type = getattr(cfg, "TASK_TYPE", "few_shot_action")
    if task_type in ["classification", "full_supervised"]:
        collate_fn = classification_collate_fn
    
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    return loader


def shuffle_dataset(loader, cur_epoch):
    sampler = loader.sampler
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(cur_epoch)


def build_dataset(cfg, split):
    """
    Build dataset based on task type.
    Args:
        cfg: config object
        split: dataset split (train/test/val)
    Returns:
        dataset object
    """
    task_type = getattr(cfg, "TASK_TYPE", "few_shot_action")
    if task_type == "classification" or task_type == "full_supervised":
        return Classification(cfg, split)
    else:
        return Few_shot(cfg, split)
