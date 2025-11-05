"""
Here we experiment with differnt LMO-based optimizers on CIFAR airbench
"""

#############################################
#                  Setup                    #
#############################################

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from optimizers import Dion, Muon, Neon, NormalizedMuon, SGDMuon, SignSGDMuon, zeropower_via_newtonschulz5, RandomNormalizedMuon, NuclearNormalizedMuon, MuonCringeMomentum
from optimizers import SpectrallyNormalizedNeon

#############################################
#               Muon optimizer              #
#############################################


# it only makes Muon much less accurate - only about 90%, do not use it
class ErrorFeedbackMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, sgd_coeff=0, error_feedback_decay=0.9):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sgd_coeff = sgd_coeff
        self.error_feedback_decay = error_feedback_decay

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                # Initialize error feedback buffer if not exists
                if "error_feedback_buffer" not in state.keys():
                    state["error_feedback_buffer"] = torch.zeros_like(g)
                
                # Initialize momentum buffer if not exists
                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                
                # Add accumulated error feedback to current gradient
                g = g + state["error_feedback_buffer"]
                
                # Apply momentum
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                # Normalize the weight
                p.data.mul_(len(p.data)**0.5 / p.data.norm())
                
                # Compute the update using zeropower approximation
                eps = 1e-12
                g_normalized = g / (g.norm() + eps)           
                update_part = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                update = (1-self.sgd_coeff) * update_part + self.sgd_coeff * g_normalized
                
                # Apply the update
                p.data.add_(update, alpha=-lr)
                
                # Compute quantization error (difference between intended and actual update)
                # The error is the difference between the original gradient and the computed update
                quantization_error = g - update
                
                # Update error feedback buffer with decay
                state["error_feedback_buffer"].mul_(self.error_feedback_decay).add_(
                    quantization_error, alpha=(1 - self.error_feedback_decay)
                )

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
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
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

        data = torch.load(data_path, map_location=torch.device('cuda'))
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {} # Saved results of image processing to be done on the first epoch
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
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get('flip', False):
                images = self.proc_images['flip'] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,)*4, 'reflect')

        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        elif self.aug.get('flip', False):
            images = self.proc_images['flip']
        else:
            images = self.proc_images['norm']
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get('flip', False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Definition             #
#############################################

# note the use of low BatchNorm stats momentum
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding='same', bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width,     widths['block1']),
            ConvGroup(widths['block1'], widths['block2']),
            ConvGroup(widths['block2'], widths['block3']),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths['block3'], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for m in model.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
        eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w) / torch.sqrt(eigenvalues.view(-1,1,1,1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    if is_head:
        print('-'*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-'*len(print_string))

logging_columns_list = ['run   ', 'epoch', 'train_acc', 'val_acc', 'tta_val_acc', 'time_seconds']
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = '{:0.4f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):

    # Test-time augmentation strategy (for tta_level=2):
    # 1. Flip/mirror the image left-to-right (50% of the time).
    # 2. Translate the image by one pixel either up-and-left or down-and-right (50% of the time,
    #    i.e. both happen 25% of the time).
    #
    # This creates 6 views per image (left/right times the two translations and no-translation),
    # which we evaluate and then weight according to the given probabilities.

    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, 'reflect')
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#                Training                  #
############################################

def main(run, model):
    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    if run == 'warmup':
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    total_train_steps = ceil(8 * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    # Create optimizers and learning rate schedulers
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if 'norm' in n and p.requires_grad]
    param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=norm_biases, lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
    
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True)#, fused=True)
    # random mix, 93.3%, 11.26 s on bs 2000 with lr=0.4, mom=0.65
    # optimizer2 = RandomNormalizedMuon(filter_params, lr=0.24, momentum=0.6, sgd_coeff=0.5, nesterov=True) # and 92.9% for bs 200
    # optimizer2 = NormalizedMuon(filter_params, lr=0.4, momentum=0.65, sgd_coeff=0.5, nesterov=True) # the best tuned F-Muon, 94.0%
    # optimizer2 = MuonCringeMomentum(filter_params, lr=0.24, momentum=0.6, nesterov=True) # it's not bad, but without Nesterov=true it does not work
    # optimizer2 = Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True) # base Muon, 94.01% 11.4 s
    switch_to_muon = True
    # optimizer2 = torch.optim.SGD(filter_params, momentum=0.85, nesterov=True, lr=0.001) # about 90%
    
    # optimizer2 = SpectrallyNormalizedNeon(filter_params, lr=0.48, momentum=0.65, nesterov=True, sgd_coeff=0) # pure Neon, 68.2%, lr is a bit high
    # optimizer2 = SpectrallyNormalizedNeon(filter_params, lr=0.3, momentum=0.6, nesterov=True, sgd_coeff=1) # Spectrally Normalized SGD, 91.4%
    # optimizer2 = NormalizedMuon(filter_params, lr=0.3, momentum=0.6, sgd_coeff=1, nesterov=True) # Frobenius NSGD, only 88%
    # optimizer2 = NormalizedMuon(filter_params, lr=1, momentum=0.6, sgd_coeff=1, nesterov=True) # Frobenius NSGD, 90.4%, mild dependency on lr
    # optimizer2= NuclearNormalizedMuon(filter_params, lr=0.4, momentum=0.6, sgd_coeff=0.5, nesterov=True) # 93.6%
    # optimizer2 = NuclearNormalizedMuon(filter_params, lr=3, momentum=0.6, sgd_coeff=1, nesterov=True) # Nuclear NSGD, 89%

    # optimizer2 = Neon(filter_params, neon_mode='kyfan', lr=0.3, momentum=0.65, nesterov=True, sgd_coeff=0) # 68.2%, lr is a bit high
    # optimizer2 = Neon(filter_params, neon_mode='kyfan', lr=0.4, momentum=0.65, nesterov=True, sgd_coeff=0) # 34%, lr is a bit high
    # optimizer2 = Neon(filter_params, neon_mode='kyfan', lr=0.45, k=5, momentum=0.65, nesterov=True, sgd_coeff=0) # 72.3%
    # optimizer2 = Neon(filter_params, neon_mode='kyfan', lr=0.45, k=20, momentum=0.65, nesterov=True, sgd_coeff=0) # 89.0%, very slow

    # optimizer2 = Neon(filter_params, neon_mode='kyfan', lr=0.45, momentum=0.65, nesterov=True, sgd_coeff=0.6) # 88.2%
    # optimizer2 = Neon(filter_params, neon_mode='fast', lr=0.45, momentum=0.65, nesterov=True, sgd_coeff=0.6) # 87.9%, must be the same as upper
    # optimizer2 = Neon(filter_params, neon_mode='kyfan', k=5, lr=0.45, momentum=0.65, nesterov=True, sgd_coeff=0.6) # 87.6%

    # optimizer2 = Dion(filter_params, lr=0.45, momentum=0.65, rank=1, momentum_decay=0.9, sgd_coeff=0) # 68.5%, but with 4.8% variance
    # optimizer2 = Dion(filter_params, lr=0.45, momentum=0.65, rank=1, momentum_decay=0.95, sgd_coeff=0) # 67.2%, but with 5% variance
    # optimizer2 = Dion(filter_params, lr=0.45, momentum=0.65, rank=1, momentum_decay=0.8, sgd_coeff=0) # 66.9%, but with 5% variance

    # optimizer2 = Dion(filter_params, lr=0.45, momentum=0.65, rank=10, momentum_decay=0.9, sgd_coeff=0) # 84.5%, with 0.4% variance
    # optimizer2 = Dion(filter_params, lr=0.45, momentum=0.65, rank=20, momentum_decay=0.9, sgd_coeff=0) # 89.3%, with 0.2% variance
    

    optimizer2 = SignSGDMuon(filter_params, lr=0.4, momentum=0.65, nesterov=True, sgd_coeff=0.5) # 94%, on par with Muon
    # optimizer2 = SGDMuon(filter_params, lr=0.24, momentum=0.6, nesterov=True, sgd_coeff=0.1) # 87% - does not work well, because it's not an LMO algorithm
    
    # optimizer2 = ErrorFeedbackMuon(filter_params, lr=0.24, momentum=0.6, nesterov=True, sgd_coeff=0, error_feedback_decay=0.9) # 89.6%, unfeasible
    
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
            group["target_momentum"] = group.get("momentum", 0)  # default to 0 if not set

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0
    def start_timer():
        starter.record()
    def stop_timer():
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0

    # Initialize the whitening layer using training images
    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    for epoch in range(ceil(total_train_steps / len(train_loader))):

        ####################
        #     Training     #
        # ####################
        # if epoch == 8 and switch_to_muon: # to go to Muon
        #     # optimizers[1] = NormalizedMuon(filter_params, lr=0.4, momentum=0.65, nesterov=True, sgd_coeff=0.5)
        #     optimizers[1].sgd_coeff = 0
        #     optimizers[1].momentum = 0.6
        #     for group in optimizers[1].param_groups:
        #         group["initial_lr"] = 0.24
        #         # group["target_momentum"] = group.get("momentum", 0)  # default to 0 if not set
        # Restart optimizers at each epoch
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
            # Reset optimizer state to clear momentum and other accumulated state
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"]
            # Clear optimizer state dict to reset momentum buffers
            opt.state.clear()
        
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction='sum').backward()
            for group in optimizers[0].param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizers[0].param_groups[1:] + optimizers[1].param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)

            '''
            Technique to change momentum, which, however, we didn't find useful
            for group in optimizer2.param_groups:
                eta = (step / total_train_steps) # percentage of training
                group["momentum"] = (group["target_momentum"] - 0.1) * eta + group["target_momentum"] * (1 - eta)
            '''
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()

        ####################
        #    Evaluation    #
        ####################

        # Save the accuracy and loss from the last training batch of the epoch
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader, tta_level=0)
        print_training_details(locals(), is_final_entry=False)
        run = None # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    epoch = 'eval'
    print_training_details(locals(), is_final_entry=True)

    return tta_val_acc

if __name__ == "__main__":

    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    # model.compile(mode='max-autotune')

    print_columns(logging_columns_list, is_head=True)
    main('warmup', model)
    accs = torch.tensor([main(run, model) for run in range(5)]) # num of repetitions in the test
    print('Mean: %.4f    Std: %.4f' % (accs.mean(), accs.std()))

    log_dir = os.path.join('logs', str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.pt')
    torch.save(dict(code=code, accs=accs), log_path)
    print(os.path.abspath(log_path))