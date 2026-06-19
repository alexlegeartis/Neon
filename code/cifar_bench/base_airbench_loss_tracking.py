"""
Here we experiment with differnt LMO-based optimizers on CIFAR airbench
"""

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
from optimizers import Dion, Muon, Neon, NormalizedMuon, SGDMuon, SignSGDMuon, zeropower_via_newtonschulz5, RandomNormalizedMuon, NuclearNormalizedMuon, MuonCringeMomentum, SignSGDNSGD
from optimizers import SpectrallyNormalizedNeon, MuonOrNSGD, MuonOrSign, RealFanion, NeonMuon, SingleDeviceNorMuon, MuonSignedUpdate
from freon_optimizers import Kaon, Freon, KaonSignedUpdate, FKaon, FFreon, FDynFreon, FreonSignedUpdate
from mlion import MLion, NLion

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

        data = torch.load(data_path, map_location=torch.device('cuda'))
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
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
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Definition             #
#############################################

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False

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
        for m in self.modules():
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

logging_columns_list = ['run   ', 'epoch', 'train_loss', 'train_acc', 'val_acc', 'tta_val_acc', 'time_seconds']
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

def main(run, model, opt_name, lr):
    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    
    num_epochs = 24
    warmup_epochs = 2

    total_train_steps = ceil(num_epochs * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))
    warmup_steps = ceil(warmup_epochs * len(train_loader))

    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if 'norm' in n and p.requires_grad]
    param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=norm_biases, lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
    
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True)
    
    if opt_name == 'Muon':
        optimizer2 = Muon(filter_params, lr=lr, momentum=0.6, nesterov=True, norm_weight=False)
    elif opt_name == 'NormalizedMuon':
        optimizer2 = NormalizedMuon(filter_params, lr=lr, momentum=0.6, sgd_coeff=0.5, nesterov=True, norm_weight=False)
    elif opt_name == 'SignSGDMuon':
        optimizer2 = SignSGDMuon(filter_params, lr=lr, momentum=0.6, nesterov=True, sgd_coeff=0.5, sign_lr_mult=0.003, norm_weight=False)
    elif opt_name == 'SignSGD':
        optimizer2 = SignSGDMuon(filter_params, lr=lr, momentum=0.6, nesterov=True, sgd_coeff=1, sign_lr_mult=0.003, norm_weight=False)
    elif opt_name == 'NSGD':
        optimizer2 = NormalizedMuon(filter_params, lr=lr, momentum=0.6, sgd_coeff=1, nesterov=True, norm_weight=False)
    elif opt_name == 'SignSGDNSGD':
        optimizer2 = SignSGDNSGD(filter_params, lr=lr, momentum=0.6, nesterov=True, sgd_coeff=0.5, sign_lr_mult=0.003, norm_weight=False)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
            group["target_momentum"] = group.get("momentum", 0)

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

    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    train_loss = 0.0 
    for epoch in range(ceil(total_train_steps / len(train_loader))):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"]
            opt.state.clear()
        
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            
            loss = F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction='sum')
            loss.backward()

            # --- Learning Rate Schedule Setup ---
            if step < warmup_steps:
                # Linear warmup from 0 to 1
                lr_multiplier = (step + 1) / warmup_steps
            else:
                # Linear decay from 1 down to 0
                lr_multiplier = (total_train_steps - step) / (total_train_steps - warmup_steps)

            # Whiten bias decays to 0 early, make sure it doesn't drop below 0 if epochs > 3
            for group in optimizers[0].param_groups[:1]:
                group["lr"] = group["initial_lr"] * max(0, 1 - step / whiten_bias_train_steps)
                
            # Rest of params and target optimizer follows the 24-epoch warmup+decay schedule
            for group in optimizers[0].param_groups[1:] + optimizers[1].param_groups:
                group["lr"] = group["initial_lr"] * lr_multiplier

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

        train_loss = F.cross_entropy(outputs.detach().float(), labels, label_smoothing=0.2, reduction='mean').item()
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader, tta_level=0)
        print_training_details(locals(), is_final_entry=False)
        run = None 

    ####################
    #  TTA Evaluation  #
    ####################

    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    epoch = 'eval'
    print_training_details(locals(), is_final_entry=True)

    return tta_val_acc, train_loss

if __name__ == "__main__":
    model = CifarNet().cuda().to(memory_format=torch.channels_last)

    num_runs = 1
    optimizers_to_test = ['SignSGDNSGD', 'NSGD', 'SignSGD', 'Muon', 'NormalizedMuon', 'SignSGDMuon']
    # optimizers_to_test = ['Muon', 'NormalizedMuon', 'NSGD']
    # lrs_to_test = [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300]
    lrs_to_test = [0.01, 0.1, 1, 10, 100]
    results_dict = {}

    for opt_name in optimizers_to_test:
        results_dict[opt_name] = {}
        for lr in lrs_to_test:
            print(f"\n{'='*60}")
            print(f"Testing Optimizer: {opt_name} | Learning Rate: {lr}")
            print(f"{'='*60}")
            print_columns(logging_columns_list, is_head=True)
            
            accs = []
            losses = []
            for run in range(num_runs):
                acc, loss = main(run, model, opt_name, lr)
                accs.append(acc)
                losses.append(loss)
                
            accs_tensor = torch.tensor(accs)
            losses_tensor = torch.tensor(losses)
            print('Overall config -> Mean Acc: %.4f (Std: %.4f) | Mean Loss: %.4f (Std: %.4f)' % (
                accs_tensor.mean(), accs_tensor.std(), losses_tensor.mean(), losses_tensor.std()))
            
            # SAVING BOTH METRICS
            results_dict[opt_name][lr] = {
                'acc': accs,
                'loss': losses
            }

    save_filename = 'lr_robustness_results.pt'
    torch.save(results_dict, save_filename)
    
    log_dir = os.path.join('logs', str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'lr_robustness_results.pt')
    torch.save(dict(code=code, results=results_dict), log_path)
    
    print(f"\nCompleted! Sweep data saved to ./_{save_filename}_ and {log_path}")