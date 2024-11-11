from typing import Callable, List, Optional, Union, Tuple

import decord
import numpy as np
import ipdb
from einops import rearrange
import torch
from torch import nn
from torch.utils.data import Dataset

import os
from torchvision.transforms import v2, InterpolationMode

import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset

import pandas as pd

class SSV2Dataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='validation', clip_len=1,
                crop_size=224, short_side_size=224, new_height=256,
                new_width=320, keep_aspect_ratio=True, num_segment=16,
                num_crop=1, test_num_segment=2, test_num_crop=3, args=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])
        if self.mode == 'train':
            self.hflip_prob = 0.5
            self.vflip_prob = 0.5
            self.scale_min = 0.5
            self.scale_max = 1.0
            self.use_random_rc = True
            self.use_cc = False
        elif self.mode == 'val':
            self.hflip_prob = 0
            self.vflip_prob = 0
            self.scale_min = 1.0
            self.scale_max = 1.0
            self.use_random_rc = False
            self.use_cc = True
        elif mode == 'test':
            self.use_random_rc = False
            self.use_cc = False
            self.data_resize =  nn.Sequential(
                v2.Resize(self.short_side_size, interpolation=InterpolationMode.BILINEAR),
            )

            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))
            self.dataset_samples = self.test_dataset

    def load_video(self, path, sample_rate_scale=1):
        buffer = self.loadvideo_decord(path, sample_rate_scale=sample_rate_scale) # T H W C
        if len(buffer) == 0:
            while len(buffer) == 0:
                warnings.warn("video {} not correctly loaded during training".format(path))
                index = np.random.randint(self.__len__())
                sample = self.dataset_samples[index]
                buffer = self.loadvideo_decord(sample, sample_rate_scale=sample_rate_scale)
        return buffer

    def __getitem__(self, index):
        sample = self.dataset_samples[index]
        scale_t = 1
        buffer = self.load_video(sample, sample_rate_scale=scale_t)
        
        if self.mode == 'train':
            label = self.label_array[index]
        elif self.mode == 'val':
            label = self.label_array[index]
            # return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0]
        elif self.mode == 'test':
            chunk_nb, split_nb = self.test_seg[index]

            buffer = torch.from_numpy(buffer.transpose(0, 3, 1, 2))
            buffer = self.data_resize(buffer)
            buffer = buffer.permute(0, 2, 3, 1).numpy()
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                / (self.test_num_crop - 1)
            temporal_start = chunk_nb # 0/1
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start::2, \
                       spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start::2, \
                       :, spatial_start:spatial_start + self.short_side_size, :]
            label = self.test_label_array[index]

        output_dict = {
            "frames": buffer,
            "label": label,
            "video_path": sample,
        }
        return output_dict


    def get_frame_inds(self, total_frames, sample_rate_scale=1):
        if self.mode == 'test':
            all_index = []
            tick = total_frames / float(self.num_segment)
            all_index = list(np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segment)] +
                               [int(tick * x) for x in range(self.num_segment)]))
            while len(all_index) < (self.num_segment * self.test_num_segment):
                all_index.append(all_index[-1])
            all_index = list(np.sort(np.array(all_index))) 
            return all_index
        else:
            average_duration = total_frames // self.num_segment
            all_index = []
            if average_duration > 0:
                all_index += list(np.multiply(list(range(self.num_segment)), average_duration) \
                                  + np.random.randint(average_duration, size=self.num_segment))
            elif total_frames > self.num_segment:
                all_index += list(np.sort(np.random.randint(total_frames, size=self.num_segment)))
            else:
                all_index += list(np.zeros((self.num_segment,)))
            all_index = list(np.array(all_index)) 

            return all_index

    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.mode in ['train', 'val']:
                vr = VideoReader(fname, 
                                 num_threads=1, 
                                 ctx=cpu(0),
                                 width=224,
                                 height=224,
                                 use_rrc=self.use_random_rc, 
                                 hflip_prob=self.hflip_prob,
                                 vflip_prob=self.vflip_prob,
                                 scale_min=self.scale_min,
                                 scale_max=self.scale_max,
                                 use_centercrop=self.use_cc)
            #if self.keep_aspect_ratio:
            else:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        all_index = self.get_frame_inds(len(vr), sample_rate_scale)        
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        return len(self.dataset_samples)



