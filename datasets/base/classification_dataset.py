#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

""" Standard classification dataset for full supervised training. """

import os
import torch
import utils.logging as logging
from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
from datasets.base.base_dataset import BaseVideoDataset
from datasets.utils.transformations import (
    KineticsResizedCropFewshot,
    ResizeToShortSide,
    RandomResizedCropVideo,
    ResizeToSize
)
from datasets.utils.random_erasing import RandomErasing

logger = logging.get_logger(__name__)


class Classification(BaseVideoDataset):
    """
    Standard classification dataset that reads from split files.
    Split files contain video filename and label separated by space.
    Format: video_filename.<ext> label (extension-agnostic; decord reads via FFmpeg, e.g. .mp4 or .webm).
    """
    
    def __init__(self, cfg, split):
        super(Classification, self).__init__(cfg, split)
        if self.split == "test" and self.cfg.PRETRAIN.ENABLE == False:
            self._pre_transformation_config_required = True

    def _get_ssl_label(self, frames):
        pass

    def _get_dataset_list_name(self):
        """
        Returns:
            dataset_list_name (string): name of the split file
        """
        if self.split == "train":
            name = getattr(self.cfg.DATA, "TRAIN_SPLIT", "Diving48_V2_train.txt")
        elif self.split == "test":
            name = getattr(self.cfg.DATA, "TEST_SPLIT", "Diving48_V2_test.txt")
        else:
            raise NotImplementedError(f"Split {self.split} not supported for classification dataset")
        
        logger.info("Reading video list from file: {}".format(name))
        return name

    def _get_sample_info(self, index):
        """
        Input: 
            index (int): video index
        Returns:
            sample_info (dict): contains different informations to be used later
                Things that must be included are:
                "path" indicating the video's path w.r.t. index
                "supervised_label" indicating the class of the video 
        """
        sample = self._samples[index]
        # Parse the line: "video_filename.<ext> label"
        parts = sample.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid line format in split file: {sample}")
        
        video_filename = parts[0]
        label = int(parts[1])
        
        # Construct full video path
        video_path = os.path.join(self.data_root_dir, video_filename)
        
        sample_info = {
            "path": video_path,
            "supervised_label": label,
        }
        return sample_info

    def _construct_dataset(self, cfg):
        """
        Constructs the dataset from split files.
        """
        if self.split in ["train", "val"]:
            self.dataset_name = self.cfg.TRAIN.DATASET
            self._num_clips = 1
        elif self.split in ["test", "submission"]:
            self.dataset_name = self.cfg.TEST.DATASET
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
        else:
            raise NotImplementedError("Split not supported")
        
        self._samples = []
        self._spatial_temporal_index = []
        dataset_list_name = self._get_dataset_list_name()

        for retry in range(5):
            try:
                logger.info("Loading {} dataset list for split '{}'...".format(self.dataset_name, self.split))
                local_file = os.path.join(cfg.OUTPUT_DIR, dataset_list_name)
                local_file = self._get_object_to_file(os.path.join(self.anno_dir, dataset_list_name), local_file)
                
                if local_file[-4:] == ".csv":
                    import pandas
                    lines = pandas.read_csv(local_file)
                    for line in lines.values.tolist():
                        for idx in range(self._num_clips):
                            self._samples.append(line)
                            self._spatial_temporal_index.append(idx)
                elif local_file[-4:] == "json":
                    import json
                    with open(local_file, "r") as f:
                        lines = json.load(f)
                    for line in lines:
                        for idx in range(self._num_clips):
                            self._samples.append(line)
                            self._spatial_temporal_index.append(idx)
                else:
                    # Read from text file: format "video_filename.mp4 label"
                    with open(local_file) as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            if line:  # Skip empty lines
                                for idx in range(self._num_clips):
                                    self._samples.append(line)
                                    self._spatial_temporal_index.append(idx)
                
                logger.info("Dataset {} split {} loaded. Length {}.".format(self.dataset_name, self.split, len(self._samples)))
                break
            except Exception as e:
                if retry < 4:
                    logger.warning(f"Retry {retry + 1}/5: {e}")
                    continue
                else:
                    raise ValueError("Data list {} not found. Error: {}".format(os.path.join(self.anno_dir, dataset_list_name), e))

        assert len(self._samples) != 0, "Empty sample list {}".format(os.path.join(self.anno_dir, dataset_list_name))

    def _config_transform(self):
        """
        Configures transforms for training and testing.
        """
        self.transform = None
        if self.split == 'train' and not self.cfg.PRETRAIN.ENABLE:
            # Training transforms
            std_transform_list = [
                transforms.ToTensorVideo(),
            ]
            
            # Add random horizontal flip if not disabled
            if not (hasattr(self.cfg.AUGMENTATION, "NO_RANDOM_FLIP") and self.cfg.AUGMENTATION.NO_RANDOM_FLIP):
                std_transform_list.append(transforms.RandomHorizontalFlipVideo())
            
            # IMPROVEMENT: Resize to short side 256 (preserving aspect ratio) then random crop
            # Reference: MMaction2 approach - better spatial diversity
            # Check if new resize approach is enabled
            use_short_side_resize = getattr(self.cfg.DATA, "USE_SHORT_SIDE_RESIZE", False)
            if use_short_side_resize:
                # New approach: Resize to short side 256 → RandomResizedCrop → Resize to 224x224
                short_side = getattr(self.cfg.DATA, "TRAIN_SHORT_SIDE", 256)
                std_transform_list.append(ResizeToShortSide(short_side))
                std_transform_list.append(RandomResizedCropVideo(size=self.cfg.DATA.TRAIN_CROP_SIZE))
            else:
                # Original approach: Resize to [256, 256] then crop
                std_transform_list.append(
                    KineticsResizedCropFewshot(
                        short_side_range=[self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                        crop_size=self.cfg.DATA.TRAIN_CROP_SIZE,
                    )
                )
            
            # Add color augmentation if enabled
            if hasattr(self.cfg.AUGMENTATION, "COLOR_AUG") and self.cfg.AUGMENTATION.COLOR_AUG:
                from datasets.utils.transformations import ColorJitter
                std_transform_list.append(
                    ColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                        gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                        is_split=getattr(self.cfg.AUGMENTATION, "IS_SPLIT", False)
                    ),
                )
            
            # Add normalization
            std_transform_list.append(
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                )
            )
            
            # Add random erasing if not disabled
            if not (hasattr(self.cfg.AUGMENTATION, "NO_RANDOM_ERASE") and self.cfg.AUGMENTATION.NO_RANDOM_ERASE):
                std_transform_list.append(RandomErasing(self.cfg))
            
            self.transform = Compose(std_transform_list)
            
        elif self.split == 'val' or self.split == 'test':
            # Test/validation transforms
            idx = -1
            if hasattr(self.cfg.DATA, "TEST_CENTER_CROP"):
                idx = self.cfg.DATA.TEST_CENTER_CROP

            if isinstance(self.cfg.DATA.TEST_SCALE, list):
                self.resize_video = KineticsResizedCropFewshot(
                    short_side_range=[self.cfg.DATA.TEST_SCALE[0], self.cfg.DATA.TEST_SCALE[1]],
                    crop_size=self.cfg.DATA.TEST_CROP_SIZE,
                    num_spatial_crops=self.cfg.TEST.NUM_SPATIAL_CROPS,
                    idx=idx
                )
            else:
                self.resize_video = KineticsResizedCropFewshot(
                    short_side_range=[self.cfg.DATA.TEST_SCALE, self.cfg.DATA.TEST_SCALE],
                    crop_size=self.cfg.DATA.TEST_CROP_SIZE,
                    num_spatial_crops=self.cfg.TEST.NUM_SPATIAL_CROPS,
                    idx=idx
                )
            std_transform_list = [
                transforms.ToTensorVideo(),
                self.resize_video,
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                )
            ]
            self.transform = Compose(std_transform_list)

    def _pre_transformation_config(self):
        """
        Set transformation parameters if required (for test time augmentation).
        """
        if hasattr(self, 'resize_video') and hasattr(self, 'spatial_idx'):
            self.resize_video.set_spatial_index(self.spatial_idx)

    def _custom_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        """
        Custom sampling method based on config SAMPLING_MODE.
        """
        if self.cfg.DATA.SAMPLING_MODE == "interval_based":
            return self._interval_based_sampling(vid_length, vid_fps, clip_idx, num_clips, num_frames, interval)
        elif self.cfg.DATA.SAMPLING_MODE == "segment_based":
            return self._segment_based_sampling(vid_length, clip_idx, num_clips, num_frames, random_sample)
        else:
            # Default to interval based
            return self._interval_based_sampling(vid_length, vid_fps, clip_idx, num_clips, num_frames, interval)

