"""
Script to run airbench_muon.py with different sgd_coeff values and plot the results.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from airbench_muon import CifarNet, CifarLoader, evaluate, BatchNorm, Conv, ConvGroup
from airbench_muon import NormalizedMuon

def create_modified_main_function(sgd_coeff):
    """
    Create a modified version of the main function with the specified sgd_coeff.
    """
    def modified_main(run, model):
        batch_size = 2000
        bias_lr = 0.053
        head_lr = 0.67
        wd = 2e-6 * batch_size

        test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
        train_loader = CifarLoader("cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
        if run == "warmup":
            # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
            train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
        total_train_steps = torch.ceil(torch.tensor(8 * len(train_loader)))
        whiten_bias_train_steps = torch.ceil(torch.tensor(3 * len(train_loader)))

        # Create optimizers and learning rate schedulers
        filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
        norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
        param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                         dict(params=norm_biases,         lr=bias_lr, weight_decay=wd/bias_lr),
                         dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
        optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=True)
        
        # Import NormalizedMuon here to avoid circular imports
        optimizer2 = NormalizedMuon(filter_params, lr=0.24, momentum=0.6, nesterov=True, sgd_coeff=sgd_coeff)
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
            ####################

            start_timer()
            model.train()
            for inputs, labels in train_loader:
                outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
                torch.nn.functional.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()
                for group in optimizer1.param_groups[:1]:
                    group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
                for group in optimizer1.param_groups[1:]+optimizer2.param_groups:
                    group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
                for group in optimizer2.param_groups:
                    eta = (step / total_train_steps) # percentage of training
                    # group["momentum"] = (group["target_momentum"] - 0.1) * eta + group["target_momentum"] * (1 - eta)
            
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
            # print_training_details(locals(), is_final_entry=False)  # Commented out to reduce output
            run = None # Only print the run number once

        ####################
        #  TTA Evaluation  #
        ####################

        start_timer()
        tta_val_acc = evaluate(model, test_loader, tta_level=2)
        stop_timer()
        epoch = "eval"
        # print_training_details(locals(), is_final_entry=True)  # Commented out to reduce output

        return tta_val_acc
    
    return modified_main
