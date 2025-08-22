#!/usr/bin/env python3
"""
Test script for Error Feedback Muon optimizers
Demonstrates the error feedback mechanism and compares with standard optimizers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from code.base_airbench import ErrorFeedbackMuon, ErrorFeedbackQuantizedMuon, NormalizedMuon

# Simple test model
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test_optimizer(optimizer_class, optimizer_kwargs, name):
    """Test an optimizer on a simple optimization problem"""
    print(f"\n=== Testing {name} ===")
    
    # Create model and data
    model = SimpleModel()
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # Create optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    
    # Training loop
    losses = []
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Loss improvement: {losses[0] - losses[-1]:.6f}")
    return losses

def main():
    print("Error Feedback Muon Optimizer Test")
    print("=" * 50)
    
    # Test different optimizers
    optimizers_to_test = [
        (NormalizedMuon, {"lr": 0.01, "momentum": 0.9, "sgd_coeff": 0.5}, "NormalizedMuon (Baseline)"),
        (ErrorFeedbackMuon, {"lr": 0.01, "momentum": 0.9, "sgd_coeff": 0.5, "error_feedback_decay": 0.9}, "ErrorFeedbackMuon"),
        (ErrorFeedbackQuantizedMuon, {"lr": 0.01, "momentum": 0.9, "sgd_coeff": 0.5, "error_feedback_decay": 0.9, "quantization_bits": 8}, "ErrorFeedbackQuantizedMuon (8-bit)"),
        (ErrorFeedbackQuantizedMuon, {"lr": 0.01, "momentum": 0.9, "sgd_coeff": 0.5, "error_feedback_decay": 0.9, "quantization_bits": 4}, "ErrorFeedbackQuantizedMuon (4-bit)"),
    ]
    
    results = {}
    for optimizer_class, kwargs, name in optimizers_to_test:
        try:
            losses = test_optimizer(optimizer_class, kwargs, name)
            results[name] = losses
        except Exception as e:
            print(f"Error testing {name}: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, losses in results.items():
        if losses:
            print(f"{name}: Final Loss = {losses[-1]:.6f}, Improvement = {losses[0] - losses[-1]:.6f}")

if __name__ == "__main__":
    main()
