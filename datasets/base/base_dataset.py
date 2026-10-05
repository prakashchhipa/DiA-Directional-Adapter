#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

""" BaseVideoDataset object to be extended for specific datasets. """

import os
import json
import shutil
import subprocess
import random
import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.utils.dlpack as dlpack
import utils.logging as logging
import re
import abc
import time
import random
import decord
import traceback
import numpy as np
from PIL import Image
from decord import VideoReader
from decord import cpu, gpu

decord.bridge.set_bridge('native')
from torchvision.transforms import Compose

import utils.bucket as bu

logger = logging.get_logger(__name__)


def _resolve_tool(name):
    """
    Locate ffmpeg/ffprobe on PATH or under CONDA_PREFIX (e.g. conda env `vifi-clip`).
    Training is expected to run with that env activated or via `conda run -n vifi-clip`.
    """
    exe = shutil.which(name)
    if exe:
        return exe
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        cand = os.path.join(prefix, "bin", name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def get_segment_duration_from_path(video_path, fps=12):
    """
    Extract segment duration from FineGym video filename.
    Format: VIDEO_E_START_END_A_CLASS1_CLASS2.mp4 or test0//videos/VIDEO_E_START_END_A_CLASS1_CLASS2.mp4
    
    Args:
        video_path (str): Path to video file
        fps (float): Frames per second (default: 12)
    
    Returns:
        duration_seconds (float): Segment duration in seconds, or None if cannot be extracted
    """
    if video_path is None:
        return None
    
    # Extract filename from path
    filename = os.path.basename(video_path)
    if not filename:
        # Try to extract from full path
        filename = video_path.split('/')[-1]
    
    # Pattern: _E_START_END_ where START and END are frame numbers
    match = re.search(r'_E_(\d+)_(\d+)_', filename)
    if match:
        start_frame = int(match.group(1))
        end_frame = int(match.group(2))
        duration_frames = end_frame - start_frame
        duration_seconds = duration_frames / fps
        return duration_seconds
    
    return None


def _parse_fraction(value, default=12.0):
    if value in (None, "", "0/0"):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if "/" in str(value):
        a, b = str(value).split("/", 1)
        try:
            return float(a) / float(b)
        except Exception:
            return default
    try:
        return float(value)
    except Exception:
        return default


def _ffprobe_video_stream(path, count_frames=False):
    """
    Return dict with width, height, fps, nb_frames (0 if unknown) via ffprobe.
    Use count_frames=True to populate nb_read_frames for WebM/VP9 when nb_frames is N/A (slower).
    """
    ffprobe = _resolve_tool("ffprobe")
    if not ffprobe:
        return None
    cmd = [ffprobe, "-v", "error"]
    if count_frames:
        cmd.append("-count_frames")
    cmd.extend(
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate",
            "-of",
            "json",
            path,
        ]
    )
    try:
        timeout = 600 if count_frames else 60
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
        streams = json.loads(out.decode("utf-8", errors="replace")).get("streams") or []
        if not streams:
            return None
        s = streams[0]
        w = int(s.get("width") or 0)
        h = int(s.get("height") or 0)
        fps = _parse_fraction(s.get("r_frame_rate") or s.get("avg_frame_rate"))
        nf = s.get("nb_frames")
        try:
            nb = int(nf) if nf not in (None, "N/A", "") else 0
        except Exception:
            nb = 0
        nrf = s.get("nb_read_frames")
        try:
            nb_read = int(nrf) if nrf not in (None, "N/A", "") else 0
        except Exception:
            nb_read = 0
        return {"width": w, "height": h, "fps": fps, "nb_frames": nb, "nb_read_frames": nb_read}
    except Exception:
        return None


def _ffprobe_format_duration_seconds(path):
    """Return container duration in seconds, or None."""
    ffprobe = _resolve_tool("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=60)
        line = out.decode("utf-8", errors="replace").strip().splitlines()
        if not line:
            return None
        return float(line[0])
    except Exception:
        return None


def _video_frame_count_and_fps(path):
    """
    Best-effort frame count and FPS for fallback decoders (ffprobe / OpenCV / PyAV).
    Some containers report inaccurate counts; callers should still clamp indices.
    """
    meta = _ffprobe_video_stream(path, count_frames=False)
    if meta and meta.get("fps", 0) > 1e-3:
        n = int(meta.get("nb_frames") or 0)
        if n <= 0:
            n = int(meta.get("nb_read_frames") or 0)
        if n > 0:
            return n, float(meta["fps"])
    if meta and meta.get("fps", 0) > 1e-3:
        dur = _ffprobe_format_duration_seconds(path)
        if dur is not None and dur > 0:
            est = int(dur * float(meta["fps"]) + 0.5)
            if est > 0:
                return est, float(meta["fps"])
    meta_slow = _ffprobe_video_stream(path, count_frames=True)
    if meta_slow and meta_slow.get("fps", 0) > 1e-3:
        n = int(meta_slow.get("nb_read_frames") or meta_slow.get("nb_frames") or 0)
        if n > 0:
            return n, float(meta_slow["fps"])
    meta = meta_slow or meta
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            if n > 0 and fps > 1e-3:
                return n, fps
    except Exception:
        pass
    try:
        import av

        with av.open(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 12.0
            n = stream.frames if stream.frames and stream.frames > 0 else 0
            if n > 0:
                return int(n), fps
    except Exception:
        pass
    if meta and meta.get("fps"):
        return 1, float(meta["fps"])
    return 1, 12.0


def _extract_frames_opencv(path, frame_indices):
    """Decode listed frame indices with OpenCV (often succeeds when decord fails on WebM/VP9)."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {path}")
    nmax = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    nmax = max(1, nmax)
    frames = []
    try:
        for idx in frame_indices:
            j = min(max(0, int(idx)), nmax - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"OpenCV failed reading frame {j} (from idx {idx}) from {path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    arr = np.stack(frames, axis=0)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _extract_frames_pyav(path, frame_indices):
    """
    Decode via PyAV by scanning forward through the stream (reliable on WebM/VP9).

    Important: frame_indices may contain duplicates (same index repeated); we must
    not dedupe. Short SSv2 clips: decode through max(index) or EOF, then clamp picks.
    """
    import av

    want_order = [int(x) for x in frame_indices]
    if not want_order:
        raise RuntimeError(f"empty frame_indices for {path}")
    max_need = max(want_order)
    frames_list = []
    container = av.open(path)
    try:
        for fi, frame in enumerate(container.decode(video=0)):
            frames_list.append(frame.to_ndarray(format="rgb24"))
            if fi >= max_need:
                break
    finally:
        container.close()
    if not frames_list:
        raise RuntimeError(f"PyAV decoded zero frames from {path}")
    n = len(frames_list)
    picked = [frames_list[min(i, n - 1)] for i in want_order]
    arr = np.stack(picked, axis=0)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _extract_frames_ffmpeg_cli(path, frame_indices):
    """Decode listed frame indices via ffmpeg CLI (works when decord/PyAV bindings misbehave)."""
    ffmpeg = _resolve_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found (install in conda env, e.g. "
            "`conda install -n vifi-clip -c conda-forge ffmpeg`, or add to PATH)"
        )
    meta = _ffprobe_video_stream(path)
    if not meta or meta.get("width", 0) <= 0 or meta.get("height", 0) <= 0:
        raise RuntimeError("ffprobe could not read video dimensions")
    w, h = int(meta["width"]), int(meta["height"])
    idxs = [int(x) for x in frame_indices]
    select_parts = "+".join([f"eq(n\\,{i})" for i in idxs])
    vf = f"select={select_parts},setpts=N/FRAME_RATE/TB"
    cmd = [
        ffmpeg,
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        path,
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"ffmpeg failed (code {proc.returncode}): {err}")
    frame_bytes = w * h * 3
    raw = proc.stdout
    if len(raw) % frame_bytes != 0:
        raise RuntimeError(
            f"ffmpeg raw length {len(raw)} not divisible by frame size {frame_bytes} for {path}"
        )
    n_out = len(raw) // frame_bytes
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((n_out, h, w, 3)).copy()
    # select may output fewer frames than requested when some indices are past EOF
    if n_out < len(idxs):
        pad = np.repeat(arr[-1:], len(idxs) - n_out, axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif n_out > len(idxs):
        arr = arr[: len(idxs)]
    return torch.from_numpy(np.ascontiguousarray(arr))


def _decode_video_frames_ffmpeg_fallback(path, frame_indices):
    """PyAV → OpenCV → ffmpeg CLI; covers WebM/VP9 when decord's threaded decoder fails."""
    errors = []
    last_exc = None
    for name, fn in (
        ("pyav", _extract_frames_pyav),
        ("opencv", _extract_frames_opencv),
        ("ffmpeg_cli", _extract_frames_ffmpeg_cli),
    ):
        try:
            return fn(path, frame_indices)
        except Exception as e:
            last_exc = e
            errors.append(f"{name}={e}")
    raise RuntimeError(
        f"Fallback decode failed for {path}; " + "; ".join(errors)
    ) from last_exc


class BaseVideoDataset(torch.utils.data.Dataset):
    """
    The BaseVideoDataset object provides a base object for all the video/image/video-text datasets.
    Abstract methods are provided for completion in the specific datasets.
    Necessary methods for all datasets such as "_decode_video", "_decode_image", 
    "__getitem__" (with standard procedure for loading the data) as well as sampling methods 
    such as "_interval_based_sampling" and "_segment_based_sampling" are implemented. 
    The specific video datasets can be extended from this dataset according to different needs.
    """

    def __init__(self, cfg, split):
        """
        For initialization of the dataset, the global cfg and the split need to provided.
        Args:
            cfg     (Config): The global config object.
            split   (str): The split, e.g., "train", "val", "test"
        """
        self.cfg = cfg
        self.split = split
        self.data_root_dir = cfg.DATA.DATA_ROOT_DIR
        self.anno_dir = cfg.DATA.ANNO_DIR

        if self.split in ["train", "val"]:
            self.dataset_name = cfg.TRAIN.DATASET
            self._num_clips = 1
        elif self.split in ["test", "submission"]:
            self.dataset_name = cfg.TEST.DATASET
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
        else:
            raise NotImplementedError("Split not supported")

        self._num_frames = cfg.DATA.NUM_INPUT_FRAMES
        self._sampling_rate = cfg.DATA.SAMPLING_RATE

        # Adaptive sampling configuration
        self._enable_adaptive_sampling = getattr(cfg.DATA, "ENABLE_ADAPTIVE_SAMPLING", False)
        self._adaptive_sampling_mode = getattr(cfg.DATA, "ADAPTIVE_SAMPLING_MODE", "rate")
        self._short_segment_threshold = getattr(cfg.DATA, "SHORT_SEGMENT_THRESHOLD", 2.0)
        self._long_segment_threshold = getattr(cfg.DATA, "LONG_SEGMENT_THRESHOLD", 6.0)
        self._short_segment_sampling_rate = getattr(cfg.DATA, "SHORT_SEGMENT_SAMPLING_RATE", 1)
        self._long_segment_sampling_rate = getattr(cfg.DATA, "LONG_SEGMENT_SAMPLING_RATE", 3)
        self._enable_temporal_augmentation = getattr(cfg.DATA, "ENABLE_TEMPORAL_AUGMENTATION", False)
        self._temporal_augmentation_prob = getattr(cfg.DATA, "TEMPORAL_AUGMENTATION_PROB", 0.5)

        self.gpu_transform = cfg.AUGMENTATION.USE_GPU  # whether or not to perform the transform on GPU
        
        # Store current video path for adaptive sampling
        self._current_video_path = None

        self.decode = self._decode_video  # decode function, decode videos by default

        self.buckets = {}

        # if set to true, _pre_transformation_config will be called before every transformations
        # this is used in the testset, where cropping positions are set for the controlled crop
        self._pre_transformation_config_required = False
        self._construct_dataset(cfg)
        self._config_transform()

    @abc.abstractmethod
    def _get_dataset_list_name(self):
        """
        Returns the list for the dataset. 
        Returns:
            name (str): name of the list to be read
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_sample_info(self, index):
        """
        Returns the sample info corresponding to the index.
        Args: 
            index (int): target index
        Returns:
            sample_info (dict): contains different informations to be used later
                Things that must be included are:
                "path" indicating the target's path w.r.t. index
                "supervised_label" indicating the class of the target 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_ssl_label(self, frames):
        """
        Uses cfg to obtain ssl label.
        Returns:
            ssl_label (dict): self-supervised labels
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _config_transform(self):
        """
        Uses cfg to config transforms and assign the transforms to self.transform
        Note: This is only used in the supervised setting.
            For self-supervised training, the augmentations are performed in the 
            corresponding generator.
        """
        self.transform = Compose([])
        raise NotImplementedError

    @abc.abstractmethod
    def _pre_transformation_config(self):
        """
            Set transformation parameters if required.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _custom_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        raise NotImplementedError

    def _get_video_frames_list(self, vid_length, vid_fps, clip_idx, random_sample=True):
        """
        Returns the list of frame indexes in the video for decoding.
        Args:
            vid_length (int): video length
            clip_idx (int): clip index, -1 if random sampling (interval based sampling)
            num_clips (int): overall number of clips for clip_idx != -1 (interval based sampling)
            num_frames (int): number of frames to sample
            interval (int): the step size for interval based sampling (interval based sampling)
            random_sample (int): whether to randomly sample one frame from each segment (segment based sampling)
        Returns:
            frame_id_list (list): indicates which frames to sample from the video
        """
        if self.cfg.PRETRAIN.ENABLE and self.split == "train":
            return self._custom_sampling(vid_length, vid_fps, clip_idx, self.cfg.TEST.NUM_ENSEMBLE_VIEWS, self._num_frames, self._sampling_rate, random_sample)
        else:
            if self.cfg.DATA.SAMPLING_MODE == "interval_based":
                # return self._interval_based_sampling(vid_length, clip_idx, self.cfg.TEST.NUM_ENSEMBLE_VIEWS, self._num_frames, self._sampling_rate)
                return self._interval_based_sampling(vid_length, vid_fps, clip_idx, self.cfg.TEST.NUM_ENSEMBLE_VIEWS, self._num_frames, self._sampling_rate)
            elif self.cfg.DATA.SAMPLING_MODE == "segment_based":
                return self._segment_based_sampling(vid_length, clip_idx, self.cfg.TEST.NUM_ENSEMBLE_VIEWS, self._num_frames, random_sample)
            else:
                raise NotImplementedError

    def _construct_dataset(self, cfg):
        """
        Constructs the dataset according to the global config object.
        Currently supports reading from csv, json and txt.
        Args:
            cfg (Config): The global config object.
        """
        self._samples = []
        self._spatial_temporal_index = []
        dataset_list_name = self._get_dataset_list_name()

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
                with open(local_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        for idx in range(self._num_clips):
                            self._samples.append(line.strip())
                            self._spatial_temporal_index.append(idx)
            logger.info("Dataset {} split {} loaded. Length {}.".format(self.dataset_name, self.split, len(self._samples)))
        except:
            raise ValueError("Data list {} not found.".format(os.path.join(self.anno_dir, dataset_list_name)))

        # validity check
        assert len(self._samples) != 0, "Empty sample list {}".format(os.path.join(self.anno_dir, dataset_list_name))

    def _read_video(self, video_path, index):
        """
        Wrapper for downloading the video and generating the VideoReader object for reading the video.
        Args:
            video_path (str): video path to read the video from. Can in OSS form or in local hard drives.
            index      (int):  for debug.
        Returns:
            vr              (VideoReader):  VideoReader object wrapping the video.
            file_to_remove  (list):         list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool):         flag for the indication of success or not.
        """
        tmp_file = str(round(time.time() * 1000)) + video_path.split('/')[-1]
        try:
            vr = None
            tmp_file = self._get_object_to_file(video_path, tmp_file, read_from_buffer=True, num_retries=1 if self.split == "train" else 20)
            vr = VideoReader(tmp_file)
            success = True
        except:
            success = False
        file_to_remove = [tmp_file] if video_path[:3] == "oss" else [None]  # if not downloaded from oss, then no files need to be removed
        return vr, file_to_remove, success

    def _decode_video(self, sample_info, index, num_clips_per_video=1):
        """
        Decodes the video given the sample info.
        Args:
            sample_info         (dict): containing the "path" key specifying the location of the video.
            index               (int):  for debug.
            num_clips_per_video (int):  number of clips to be decoded from each video. set to 2 for contrastive learning and 1 for others.
        Returns:
            data            (dict): key "video" for the video data.
            file_to_remove  (list): list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool): flag for the indication of success or not.
        """
        path = sample_info["path"]
        # Store current video path for adaptive sampling
        self._current_video_path = path

        if self.split == "train":
            clip_idx = -1
            self.spatial_idx = -1
        elif self.split == "val":
            clip_idx = -1
            self.spatial_idx = 0
        elif self.split == "test" or self.split == "submission":
            clip_idx = self._spatial_temporal_index[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                self.spatial_idx = 0
            else:
                self.spatial_idx = self._spatial_temporal_index[index] % self.cfg.TEST.NUM_SPATIAL_CROPS

        # decord's threaded VP9/WebM path often hits AVERROR(EAGAIN) (-11); skip it for local .webm
        # and decode with PyAV/ffmpeg instead (no per-frame warning spam). OSS keeps decord+retry.
        skip_decord_webm = (
            getattr(self.cfg.DATA, "SKIP_DECORD_FOR_WEBM", True)
            and not str(path).startswith("oss")
            and os.path.basename(path).lower().endswith(".webm")
        )
        if skip_decord_webm:
            file_to_remove = [None]
            meta_n, meta_fps = _video_frame_count_and_fps(path)
            meta_n = max(1, meta_n)
            frame_list = []
            for idx in range(num_clips_per_video):
                list_ = self._get_video_frames_list(
                    meta_n,
                    meta_fps,
                    clip_idx,
                    random_sample=True if self.split == "train" else False,
                )
                list_ = [min(max(0, int(i)), meta_n - 1) for i in list_]
                frames = _decode_video_frames_ffmpeg_fallback(path, list_)
                frame_list.append(frames)
            frames = torch.stack(frame_list)
            if num_clips_per_video == 1:
                frames = frames.squeeze(0)
            return {"video": frames}, file_to_remove, True

        vr, file_to_remove, success = self._read_video(path, index)

        if not success:
            return vr, file_to_remove, success

        frame_list = []
        decord_ok = True
        try:
            for idx in range(num_clips_per_video):
                # for each clip in the video,
                # a list is generated before decoding the specified frames from the video
                list_ = self._get_video_frames_list(
                    len(vr),
                    vr.get_avg_fps(),
                    clip_idx,
                    random_sample=True if self.split == "train" else False
                )
                frames = dlpack.from_dlpack(vr.get_batch(list_).to_dlpack()).clone()
                frame_list.append(frames)
        except Exception as e:
            decord_ok = False
            logger.debug(
                "decord decode failed for %s (%s); using PyAV/OpenCV/ffmpeg CLI fallback",
                path,
                e,
            )
        finally:
            del vr

        if not decord_ok:
            meta_n, meta_fps = _video_frame_count_and_fps(path)
            meta_n = max(1, meta_n)
            frame_list = []
            for idx in range(num_clips_per_video):
                list_ = self._get_video_frames_list(
                    meta_n,
                    meta_fps,
                    clip_idx,
                    random_sample=True if self.split == "train" else False
                )
                list_ = [min(max(0, int(i)), meta_n - 1) for i in list_]
                frames = _decode_video_frames_ffmpeg_fallback(path, list_)
                frame_list.append(frames)

        frames = torch.stack(frame_list)
        if num_clips_per_video == 1:
            frames = frames.squeeze(0)
        return {"video": frames}, file_to_remove, True

    def _read_image(self, path, index):
        """
        Wrapper for downloading the image and generating the PIL.Image object for reading the image.
        Args:
            path    (str): image path to read the image from. Can in OSS form or in local hard drives.
            index   (int):  for debug.
        Returns:
            img             (PIL.Image):    image object for further processing.
            file_to_remove  (list):         list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool):         flag for the indication of success or not.
        """
        tmp_file = str(round(time.time() * 1000)) + path.split('/')[-1]
        for tmp in range(10):
            try:
                img = None
                tmp_file = self._get_object_to_file(path, tmp_file, read_from_buffer=True)
                if isinstance(tmp_file, str):
                    with open(tmp_file, 'rb') as f:
                        img = Image.open(f).convert('RGB')
                else:
                    img = Image.open(tmp_file).convert('RGB')
                success = True
                break
            except:
                success = False
        file_to_remove = [tmp_file] if path[:3] == "oss" else [None]
        return img, file_to_remove, success

    def _decode_image(self, sample_info, index, num_clips_per_video=1):
        """
        Decodes the image given the sample info.
        Args:
            sample_info         (dict): containing the "path" key specifying the location of the image.
            index               (int):  for debug.
            num_clips_per_video (int):  number of clips to be decoded from each video. set to 2 for contrastive learning and 1 for others.
                                        specifically in this function, num_clips_per_video does not matter since all things to be decoded is one image.
        Returns:
            data            (dict): key "video" for the image data.
                                    because this is a video database, the images will be in the shape of 1,H,W,C before further processing.
            file_to_remove  (list): list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool): flag for the indication of success or not.
        """
        path = sample_info["path"]
        img, tmp_file, success = self._read_image(path, index)

        if not success:
            return None, tmp_file, success

        frame = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(img.size[1], img.size[0], len(img.getbands()))
        frame = frame.unsqueeze(0)  # 1, H, W, C
        return {"video": frame}, tmp_file, True

    def __getitem__(self, index):
        """
        Gets the specified data.
        Args:
            index (int): the index of the data in the self._samples list.
        Returns:
            frames (dict): {
                "video": (tensor),
                "text_embedding" (optional): (tensor)
            }
            labels (dict): {
                "supervised": (tensor),
                "self-supervised" (optional): (...)
            }
        """
        sample_info = self._get_sample_info(index)

        # decode the data
        retries = 1 if self.split == "train" else 10
        for retry in range(retries):
            try:
                data, file_to_remove, success = self.decode(
                    sample_info, index, num_clips_per_video=self.num_clips_per_video if hasattr(self, 'num_clips_per_video') else 1
                )
                break
            except Exception as e:
                success = False
                traceback.print_exc()
                logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                    retry + 1, retries, index, sample_info["path"]
                ))

        if not success:
            return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)

        if self.gpu_transform:
            for k, v in data.items():
                data[k] = v.cuda(non_blocking=True)
        if self._pre_transformation_config_required:
            self._pre_transformation_config()

        labels = {}
        labels["supervised"] = sample_info["supervised_label"] if "supervised_label" in sample_info.keys() else {}
        if self.cfg.PRETRAIN.ENABLE:
            # generates the different augmented samples for pre-training
            try:
                data, labels["self-supervised"] = self.ssl_generator(data, index)
            except Exception as e:
                traceback.print_exc()
                print("Error at Vid index: {}, Vid path: {}, Vid shape: {}".format(
                    index, sample_info["path"], data["video"].shape
                ))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
        else:
            # augment the samples for supervised training
            labels["self-supervised"] = {}
            if "flow" in data.keys() and "video" in data.keys():
                data = self.transform(data)
            elif "video" in data.keys():
                data["video"] = self.transform(data["video"])  # C, T, H, W = 3, 16, 240, 320, RGB

        # if the model is SlowFast, generate two sets of inputs with different framerates.
        if self.cfg.VIDEO.BACKBONE.META_ARCH == "Slowfast":
            slow_idx = torch.linspace(0, data["video"].shape[1], data["video"].shape[1] // self.cfg.VIDEO.BACKBONE.SLOWFAST.ALPHA + 1).long()[:-1]
            fast_frames = data["video"].clone()
            slow_frames = data["video"][:, slow_idx, :, :].clone()
            data["video"] = [slow_frames, fast_frames]
        bu.clear_tmp_file(file_to_remove)

        return data, labels, index, {}

    def _get_object_to_file(self, obj_file: str, local_file, read_from_buffer=False, num_retries=10):
        """
        Wrapper for downloading the file object.
        Args:
            obj_file         (str):  the target file to be downloaded (if it starts by "oss").
            local_file       (str):  the local file to store the downloaded file.
            read_from_butter (bool): whether or not to directly download to the memory
            num_retries      (int):  number of retries.
        Returns:
            str or BytesIO depending on the read_from_buffer flag
            if read_from_buffer==True:
                returns BytesIO
            else:
                returns str (indicating the location of the specified file)
        """
        if obj_file[:3] == "oss":
            bucket_name = obj_file.split('/')[2]
            if bucket_name not in self.buckets.keys():
                self.buckets[bucket_name] = self._initialize_oss(bucket_name)
            if read_from_buffer:
                local_file = bu.read_from_buffer(
                    self.buckets[bucket_name],
                    obj_file,
                    bucket_name,
                    num_retries
                )
            else:
                bu.read_from_bucket(
                    self.buckets[bucket_name],
                    obj_file,
                    local_file,
                    bucket_name,
                    num_retries
                )
            return local_file
        else:
            return obj_file

    def _initialize_oss(self, bucket_name):
        """
        Initializes the oss bucket.
        Currently supporting two OSS accounts.
        """
        if hasattr(self.cfg.OSS, "SECONDARY_DATA_OSS") and \
                self.cfg.OSS.SECONDARY_DATA_OSS.ENABLE and \
                bucket_name in self.cfg.OSS.SECONDARY_DATA_OSS.BUCKETS:
            return bu.initialize_bucket(
                self.cfg.OSS.SECONDARY_DATA_OSS.KEY,
                self.cfg.OSS.SECONDARY_DATA_OSS.SECRET,
                self.cfg.OSS.SECONDARY_DATA_OSS.ENDPOINT,
                bucket_name
            )
        else:
            return bu.initialize_bucket(
                self.cfg.OSS.KEY,
                self.cfg.OSS.SECRET,
                self.cfg.OSS.ENDPOINT,
                bucket_name
            )

    def __len__(self):
        """
        Returns the number of samples.
        """
        if hasattr(self.cfg.TRAIN, "NUM_SAMPLES") and self.split == 'train':
            return self.cfg.TRAIN.NUM_SAMPLES
        else:
            return len(self._samples)


    # def _interval_based_sampling(self, vid_length, clip_idx, num_clips, num_frames, interval):
    def _interval_based_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval):
        """
        Interval-based sampling with adaptive sampling rate support.
        Adapts sampling rate based on segment duration for mixed duration clips.
        """
        if num_frames == 1:
            index = [random.randint(0, vid_length - 1)]
        else:
            # First, check for special sampling rate configurations
            if self.split == "train" and hasattr(self.cfg.DATA, "SAMPLING_RATE_TRAIN"):
                interval = self.cfg.DATA.SAMPLING_RATE_TRAIN
                clip_length = num_frames * interval * vid_fps / self.cfg.DATA.TARGET_FPS
            elif hasattr(self.cfg.DATA, "SAMPLING_RATE_TEST") and self.cfg.DATA.SAMPLING_RATE_TEST > 40:
                interval = vid_length // num_frames
                clip_length = vid_length // num_frames * num_frames
                index = [random.randint(ind * interval, ind * interval + interval - 1) for ind in range(num_frames)]
                return index
            elif self.cfg.DATA.SAMPLING_RATE > 40:  # SAMPLING_RATE_TEST
                interval = vid_length // num_frames
                clip_length = vid_length // num_frames * num_frames
                index = [random.randint(ind * interval, ind * interval + interval - 1) for ind in range(num_frames)]
                return index
            else:
                # Apply adaptive sampling rate if enabled (for default case)
                if self._enable_adaptive_sampling and self._adaptive_sampling_mode == "rate":
                    segment_duration = get_segment_duration_from_path(self._current_video_path, vid_fps)
                    if segment_duration is not None:
                        if segment_duration < self._short_segment_threshold:
                            # Short segment: use dense sampling to capture all frames
                            interval = self._short_segment_sampling_rate
                            if self.split == "train":
                                logger.debug(f"Short segment ({segment_duration:.2f}s): using sampling_rate={interval}")
                        elif segment_duration >= self._long_segment_threshold:
                            # Long segment: use sparse sampling to cover more temporal range
                            interval = self._long_segment_sampling_rate
                            if self.split == "train":
                                logger.debug(f"Long segment ({segment_duration:.2f}s): using sampling_rate={interval}")
                        # Medium segments use default interval (already set)
                # transform FPS
                clip_length = num_frames * interval * vid_fps / self.cfg.DATA.TARGET_FPS

            if clip_length > vid_length:
                clip_length = vid_length // num_frames * num_frames

            max_idx = max(vid_length - clip_length + 1, 0)
            
            # Temporal augmentation for long segments during training
            if (self.split == "train" and 
                self._enable_temporal_augmentation and 
                max_idx > 0):
                segment_duration = get_segment_duration_from_path(self._current_video_path, vid_fps)
                if (segment_duration is not None and 
                    segment_duration >= self._long_segment_threshold and
                    random.random() < self._temporal_augmentation_prob):
                    # Randomly shift start position within segment for temporal augmentation
                    start_idx = random.uniform(0, max_idx)
                elif clip_idx == -1:  # random sampling
                    start_idx = random.uniform(0, max_idx)
                else:
                    if num_clips == 1:
                        start_idx = max_idx / 2
                    else:
                        start_idx = max_idx * clip_idx / num_clips
            else:
                if clip_idx == -1:  # random sampling
                    start_idx = random.uniform(0, max_idx)
                else:
                    if num_clips == 1:
                        start_idx = max_idx / 2
                    else:
                        start_idx = max_idx * clip_idx / num_clips
            
            end_idx = start_idx + clip_length - interval

            index = torch.linspace(start_idx, end_idx, num_frames)
            index = torch.clamp(index, 0, vid_length - 1).long()

        return index

    def _segment_based_sampling(self, vid_length, clip_idx, num_clips, num_frames, random_sample):
        """
        Generates the frame index list using segment based sampling.
        Args:
            vid_length    (int):  the length of the whole video (valid selection range).
            clip_idx      (int):  -1 for random temporal sampling, and positive values for sampling specific clip from the video
            num_clips     (int):  the total clips to be sampled from each video. 
                                    combined with clip_idx, the sampled video is the "clip_idx-th" video from "num_clips" videos.
            num_frames    (int):  number of frames in each sampled clips.
            random_sample (bool): whether or not to randomly sample from each segment. True for train and False for test.
        Returns:
            index (tensor): the sampled frame indexes
        """
        index = torch.zeros(num_frames)
        index_range = torch.linspace(0, vid_length, num_frames + 1)
        for idx in range(num_frames):
            if random_sample:
                index[idx] = random.uniform(index_range[idx], index_range[idx + 1])
            else:
                if num_clips == 1:
                    index[idx] = (index_range[idx] + index_range[idx + 1]) / 2
                else:
                    index[idx] = index_range[idx] + (index_range[idx + 1] - index_range[idx]) * (clip_idx + 1) / num_clips
        index = torch.round(torch.clamp(index, 0, vid_length - 1)).long()

        return index
