#############################################
#                DataLoader                 #
#############################################
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from config import device

from math import ceil
import os

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device=device)
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616), device=device)

#@torch.compile(mode='max-autotune')
def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

#@torch.compile(mode='max-autotune')
def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location='cpu')
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        self.images = (self.images.float() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN.cpu(), CIFAR_STD.cpu())
        self.proc_images = {}
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images['norm'] = self.normalize(self.images)
            if self.aug.get('flip', False):
                images = self.proc_images['flip'] = batch_flip_lr(images)
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,)*4, 'reflect')
        self.epoch += 1
        indices = torch.randperm(len(self.images)) if self.shuffle else torch.arange(len(self.images))
        for i in range(0, len(self.images) - self.batch_size + 1 if self.drop_last else len(self.images), self.batch_size):
            idx = indices[i:i+self.batch_size]
            images = self.proc_images['norm'][idx]
            if self.aug.get('flip', False):
                images = batch_flip_lr(images)
            if self.aug.get('translate', 0) > 0:
                images = batch_crop(self.proc_images['pad'][idx], 32)
            yield images.to(device), self.labels[idx].to(device)

