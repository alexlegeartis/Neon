"""
neon_light.py
A lightweight version of the Muon optimizer using a perceptron model
Runs on AMD Radeon GPUs using ROCm
"""
from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "9.0.0")

import os
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from math import ceil

# Enable ROCm backend
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############################################
#               Muon optimizer              #
#############################################

def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """Simplified Newton-Schulz iteration for whitening"""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if 'momentum_buffer' not in state.keys():
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf

                p.data.mul_(len(p.data)**0.5 / p.data.norm())
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                p.data.add_(update, alpha=-lr)

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

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

        data = torch.load(data_path, map_location='cpu')  # Load to CPU first
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        # Convert to float32 instead of half
        self.images = (self.images.float() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
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

        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        elif self.aug.get('flip', False):
            images = self.proc_images['flip']
        else:
            images = self.proc_images['norm']
        if self.aug.get('flip', False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device='cpu')
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs].to(device), self.labels[idxs].to(device))

#############################################
#            Network Definition             #
#############################################

class SimplePerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32*32*3, 512)
        self.linear2 = nn.Linear(512, 10)
        self.activ = nn.GELU()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activ(x)
        x = self.linear2(x)
        return x

############################################
#                Training                  #
############################################

def main(run, model):
    batch_size = 128
    lr = 0.01
    momentum = 0.9
    total_epochs = 10

    train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    test_loader = CifarLoader('cifar10', train=False, batch_size=batch_size)

    optimizer = Muon(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    
    for epoch in range(total_epochs):
        print(f"Epoch {epoch+1}/{total_epochs}")
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        print(f"Train Loss: {total_loss/len(train_loader):.3f} | Train Acc: {train_acc:.3f}%")
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        print(f"Test Acc: {test_acc:.3f}%")
        
    return test_acc

if __name__ == "__main__":
    model = SimplePerceptron().to(device)
    print("Starting training...")
    final_acc = main(0, model)
    print(f"Final test accuracy: {final_acc:.2f}%") 