import os
from typing import Callable, List, Optional, Union, Tuple

import decord
import numpy as np
import ipdb
import PIL

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2, InterpolationMode, RandAugment

#from pytorchvideo.transforms import RandAugment

import src.utils.functional as FF

class KineticsDataset(Dataset):
    """Core class for loading video datasets.

    Currently does only CenterCropping (validation set) for simpler implementation.
    TODO: add in robust impl of spatial crop? 
    """

    def __init__(self, metadata, mode='train'):
        print("Setting up dataset...")
        self.data_list = self.load_data_list(metadata)
        # HACK TO ENSURE IT WORKS ON 8 GPUS: FIX THIS LATER
        remainder = len(self.data_list) % 8
        self.data_list = self.data_list[:-remainder]
        assert len(self.data_list) % 8 == 0
        # Move this to a config.
        self.num_decode_threads = 1
        self.clip_len = 16
        self.frame_interval = 4
        # TEMPORARY: WE NEED TO REMOVE THIS, AFTER RESIZING DOWN TO 224.
        self.size = (224, 224)
        self.scale = (-1, 224)
        self.patch_size = (16, 16)
        self.tubelet_size = 2
        self.transform = None
        self.mode = mode
        # ipdb.set_trace()
        self.rand_aug = RandAugment(
            magnitude=7, num_ops=4
        )
        
        if mode == 'train':
            #ipdb.set_trace()
            self.train = True
            self.use_random_rc = True
            self.num_clips = 1
            self.use_cc = False
            self.num_spatial_crops = 1
        elif mode == 'val': 
            self.train = False
            self.use_random_rc = False
            self.num_clips = 1
            self.use_cc = True # use resized 224.
            self.num_spatial_crops = 1
        elif mode == 'test':
            self.train = False
            self.use_random_rc = False
            self.num_clips = 4
            self.use_cc = False
            self.num_spatial_crops = 3
            self.clip_idx = []
            # Need to use transform in here to handle this, 
            # since we have to do spatial cropping too.
            self.transform = nn.Sequential(
                #v2.ToTensor(),
                v2.Resize(224, interpolation=InterpolationMode.BICUBIC),
                #v2.CenterCrop(self.size),
            )
            # new_data = []
            # for (path, label) in self.data_list:
            #     for i in range(self.num_clips):
            #         new_data.append((path, label))
            #         self.clip_idx.append(i)
            # self.data_list = new_data

        self.log_ids = set(random.choices(range(len(self.data_list)), k=10))
        print("Done setting up dataset.")

    def __len__(self):
        return len(self.data_list)

    def load_data_list(self, metadata: str) -> List[str]:
        """
        Loads the data list from the metadata file. The metadata
        file must be formatted as follows:

        /path/to/video1 label1
        /path/to/video2 label2

        :param metadata: The path to the metadata file.
        :return: A list of video paths

        # TODO include number of frames here too to match other repos.
        """
        data_list = []
        with open(metadata) as f:
            for line in f:
                line = line.strip()
                # Append video_path, label
                data_list.append(line.split())
        return data_list


    def _three_crop(self, clip):
        """Does multiple spatial cropping. 
        Basically crops the frames along the longer edge into 3
        equally spaced NxN crops. instead of Nxlonger side.

        Not currently enabled, but will do so soon.
        
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        #ipdb.set_trace()
        h, w = self.size
        # if isinstance(clip[0], np.ndarray):
        #     im_h, im_w, im_c = clip[0].shape
        # elif isinstance(clip[0], PIL.Image.Image):
        #     im_w, im_h = clip[0].size
        # else:
        #     raise TypeError('Expected numpy.ndarray or PIL.Image' +
        #                     'but got list of {0}'.format(type(clip[0])))
        # if w != im_w and h != im_h:
        #     clip = FF.resize_clip(clip, self.size, interpolation="bilinear")
        #     im_h, im_w, im_c = clip[0].shape
        im_h, im_w, im_c = clip[0].shape
        step = np.max((np.max((im_w, im_h)) - self.size[0]) // 2, 0)
        cropped = []
        for i in range(3):
            if (im_h > self.size[0]):
                x1 = 0
                y1 = i * step
                cropped.extend(FF.crop_clip(clip, y1, x1, h, w))
            else:
                x1 = i * step
                y1 = 0
                cropped.extend(FF.crop_clip(clip, y1, x1, h, w))
        
        cropped = np.array(cropped)
        #ipdb.set_trace()
        return cropped

    def _get_frame_inds(self, total_frames: int) -> np.ndarray:
        """Sample frames from the video.

        Code adapted from the VideoMAE frame sampling.

        This isn't deterministic though!!!
        """
        # CHECK FRAME INTERVAL.
        SAMPLE_RATE_SCALE = 1
        converted_len = int(self.clip_len * self.frame_interval)
        seg_len = total_frames // self.num_clips
        all_index = []
        for i in range(self.num_clips):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_interval)
                index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_interval) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i*seg_len
            all_index.append(list(index[::int(SAMPLE_RATE_SCALE)]))

            
        return np.array(all_index)

    def __getitem__(self, idx):
        """
        This loader is light on purpose to free up CPU work for decoding videos, 
        the rest of the work occurs on GPU.
        """
        video_path, label = self.data_list[idx]
        # TODO: remove this line. 
        if not os.path.exists(video_path):
            return dict()
        # MOve this to a config.
        if self.use_random_rc:
            hflip_prob=0.5
            vflip_prob=0.5
            scale_min=0.5
            scale_max=1.0
        else:
            hflip_prob = 0
            vflip_prob = 0
            scale_min = 1.0
            scale_max = 1.0

        # During training time, we want to efficiently extract a random crop. 
        # At validation and test, we want a center crop from the 224p version, 
        # resized first. So, we need to not do the fused version, or we need
        # to resize down to 224.
        if self.mode in ['train', 'val']: 
            container = decord.VideoReader(video_path, 
                                        num_threads=self.num_decode_threads,
                                        width=224,
                                        height=224,
                                        use_rrc=self.use_random_rc,
                                        hflip_prob=hflip_prob,
                                        vflip_prob=vflip_prob,
                                        scale_min=scale_min,
                                        scale_max=scale_max,
                                        use_centercrop=self.use_cc)
        else:
           container = decord.VideoReader(video_path,
                                          num_threads=self.num_decode_threads)

        frame_inds = self._get_frame_inds(total_frames=len(container))
        # if self.mode == 'test':
        #     #ipdb.set_trace()
        #     assert len(frame_inds) == self.num_clips
        #     frame_inds = frame_inds[self.clip_idx[idx]]
        imgs = container.get_batch(frame_inds).asnumpy()
        
        # if self.mode == 'train':
        #     #ipdb.set_trace()
        #     imgs = imgs.transpose(0, 3, 1, 2) # T H W C -> T C H W
        #     imgs = torch.from_numpy(imgs)
        #     imgs = self.rand_aug(imgs)
        #     imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()

        if self.mode == 'test':
            #ipdb.set_trace()
            imgs = imgs.transpose(0, 3, 1, 2)
            imgs = torch.from_numpy(imgs)
            imgs = self.transform(imgs)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            if self.num_spatial_crops > 1:
                 imgs = self._three_crop(imgs)
    
        output_dict = {
            "frames": imgs,
            "label": int(label),
            "video_path": video_path
        }

        return output_dict


if __name__ == "__main__":
    _ = KineticsDataset()
