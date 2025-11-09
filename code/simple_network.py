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
import math
torch.backends.cudnn.benchmark = True

from optimizers import NormalizedMuon

def zeropower_via_newtonschulz5(G, steps=3, eps=1e-12):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G # .bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
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
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                # norm = p.data.norm()
                # if norm < 1e-10:
                #     norm = 1e-10
                #     print("Crisis")
                # p.data.mul_(len(p.data)**0.5 / norm) # normalize the weight -- it's a crutch
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                p.data.add_(update, alpha=-lr) # take a step



#############################################
#                DataLoader                 #
#############################################

# Use plain tuples for mean/std to avoid dtype/device confusion.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

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
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)

        # load onto GPU (map_location) to match rest of code
        data = torch.load(data_path, map_location=torch.device("cuda"))
        self.images, self.labels, self.classes = data["images"], data["labels"], data["classes"]
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        # **CHANGE**: use float() not half() to keep dtype consistent with model weights (float32).
        self.images = (self.images.float() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {} # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate"], "Unrecognized key: %s" % k

        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):

        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,)*4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get("flip", False):
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
# Model configurations

MODEL_CONFIGS = {
    'simple_plus': {
        'input_size': 32 * 32 * 3,
        'hidden_sizes': [2048, 1024, 512],
        'output_size': 10,
        'dropout': 0.1
    },
}

class MLPBlock(nn.Module):
    """Linear -> GELU -> Dropout"""
    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ImprovedPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = MODEL_CONFIGS['simple_plus']

        layers = []
        input_dim = cfg['input_size']
        for hidden_dim in cfg['hidden_sizes']:
            layers.append(MLPBlock(input_dim, hidden_dim, dropout=cfg['dropout']))
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim, cfg['output_size'])
        self._init_head()

    def _init_head(self):
        nn.init.kaiming_normal_(self.head.weight, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.backbone(x)
        x = self.head(x)
        return x
    def reset(self):
        """Reinitialize all model parameters."""
        def weight_reset(m):
            if isinstance(m, (nn.Linear,)):
                m.reset_parameters()
            elif hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.apply(weight_reset)

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ""
    for col in columns_list:
        print_string += "|  %s  " % col
    print_string += "|"
    if is_head:
        print("-"*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-"*len(print_string))

logging_columns_list = ["run   ", "epoch", "train_acc", "val_acc", "tta_val_acc", "time_seconds"]
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = "{:0.4f}".format(var)
        else:
            assert var is None
            res = ""
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
        padded_inputs = F.pad(inputs, (pad,)*4, "reflect")
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

def main(run, model):

    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader("cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    if run == "warmup":
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    total_train_steps = ceil(16 * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    # optimizer2 = NormalizedMuon(model.parameters(), lr=0.1, momentum=0.6, nesterov=True, sgd_coeff=0)
    optimizer2 = Muon(model.parameters(), lr=0.1, momentum=0.6, nesterov=True)
    
    # optimizer2 = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizers = [optimizer2]
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
    # model.init_whiten(train_images)
    stop_timer()

    for epoch in range(ceil(total_train_steps / len(train_loader))):

        ####################
        #     Training     #
        ####################

        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs) # , whiten_bias_grad=(step < whiten_bias_train_steps))
            F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()
            for group in optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
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
    epoch = "eval"
    print_training_details(locals(), is_final_entry=True)

    return tta_val_acc

if __name__ == "__main__":

    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    model = ImprovedPerceptron().cuda().to(memory_format=torch.channels_last)
    # model.compile(mode="max-autotune")

    print_columns(logging_columns_list, is_head=True)
    # main("warmup", model)
    accs = torch.tensor([main(run, model) for run in range(200)])
    print("Mean: %.4f    Std: %.4f" % (accs.mean(), accs.std()))

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, accs=accs), log_path)
    print(os.path.abspath(log_path))
