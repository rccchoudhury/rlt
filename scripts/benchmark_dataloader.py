"""
A script to measure the throughput of loading videos
Can vary parameters in the process to measure stuff. 
"""

import argparse
import time
import ipdb
import sys

sys.path.append("..")
from collections import Counter
import numpy as np
from pytorchvideo.transforms import RandAugment
from tqdm import tqdm
import torch
#torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2

from src.data.kinetics_dataset import KineticsDataset
from src.data.kinetics_vmae import VideoClsDataset
from src.models.mixup import Mixup
from src.models.random_erasing import RandomErasing
from src.models.tokenizer import Tokenizer


def main(args):
    metadata_path = args.metadata_path
    dataset = KineticsDataset(metadata=metadata_path, mode='train')
    #assert dataset.use_random_rc
    #dataset.num_clips = 1
    num_videos = len(dataset)
    subset = Subset(dataset=dataset, indices=list(range(0, num_videos)))
    dataloader = DataLoader(subset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    transform = torch.nn.Sequential(
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        )
    
    mixup_fn = Mixup(mixup_alpha=0.0,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=400)
    
    random_erase_fn = RandomErasing(
            probability=0.15,
            mode='const',
            device='cuda',
            min_count=1,
        )
    
    #rand_aug_fn = RandAugment(num_layers=4, magnitude=7)
    tokenizer = Tokenizer(drop_policy='none',
                            drop_param=0.05,
                            transform=transform,
                            mixup_fn=None,
                            random_erase_fn=None,
                            rand_aug_fn=None)
    tokenizer.cuda()
    fracs = []
    label_counter = Counter()
    start_time = time.time()
    for batch in tqdm(dataloader, total=len(dataloader), desc='Benchmarking dataloader', colour='green'):
        frames = batch["frames"].cuda()
        targets = batch["label"].cuda()
        all_labels = targets.flatten().tolist()
        label_counter.update(all_labels)

        with torch.no_grad():
            output_dict = tokenizer(frames, targets, is_training=True)
            num_tokens = output_dict["num_tokens"].detach()
            frac_retained = num_tokens.sum() / (1568 * args.batch_size)
            fracs.append(frac_retained.cpu().numpy())
    print("Time taken: %0.2f seconds" % (time.time() - start_time))
    print("Throughput: %0.2f clips / second" % (num_videos / (time.time() - start_time)))
    for idx in range(400):
        print("{}: {}".format(idx, label_counter[idx]))
    fracs = np.array(fracs)
    print("Mean: %0.3f" % fracs.mean())
    print("Std: %0.3f" % fracs.std())
    print("Max: %0.3f" % fracs.max())
    print("Min: %0.3f" % fracs.min())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=12)

    args = parser.parse_args()
    main(args)