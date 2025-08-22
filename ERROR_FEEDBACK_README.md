# Error Feedback Muon Optimizers

This document describes the new Error Feedback optimizers added to the base_airbench.py file, which implement error feedback mechanisms to improve optimization stability and handle gradient quantization.

## Overview

Error Feedback is a technique that accumulates quantization or approximation errors and adds them back to gradients in subsequent optimization steps. This helps maintain convergence quality even when gradients are compressed or approximated.

## New Optimizer Classes

### 1. ErrorFeedbackMuon

A basic error feedback implementation that accumulates optimization errors and compensates for them in future steps.

**Key Features:**
- Accumulates the difference between intended and actual parameter updates
- Applies error feedback with configurable decay rate
- Maintains the same interface as other Muon optimizers

**Parameters:**
- `lr`: Learning rate
- `momentum`: Momentum coefficient
- `nesterov`: Whether to use Nesterov momentum
- `sgd_coeff`: Coefficient for SGD component (0 = pure Muon, 1 = pure SGD)
- `error_feedback_decay`: Decay rate for error feedback buffer (default: 0.9)

**Usage:**
```python
optimizer = ErrorFeedbackMuon(
    model.parameters(), 
    lr=0.45, 
    momentum=0.65, 
    nesterov=True, 
    sgd_coeff=0.6, 
    error_feedback_decay=0.9
)
```

### 2. ErrorFeedbackQuantizedMuon

An advanced error feedback implementation that handles actual gradient quantization, making it suitable for distributed training with compressed gradients.

**Key Features:**
- Implements actual gradient quantization to specified bit precision
- Accumulates quantization errors and feeds them back
- Configurable quantization bits (4, 8, 16, 32)
- Useful for communication-efficient distributed training

**Parameters:**
- All parameters from ErrorFeedbackMuon
- `quantization_bits`: Number of bits for gradient quantization (default: 8)

**Usage:**
```python
optimizer = ErrorFeedbackQuantizedMuon(
    model.parameters(), 
    lr=0.45, 
    momentum=0.65, 
    nesterov=True, 
    sgd_coeff=0.6, 
    error_feedback_decay=0.9,
    quantization_bits=8
)
```

## How Error Feedback Works

1. **Error Accumulation**: The optimizer computes the difference between the intended update and the actual (potentially quantized) update
2. **Error Storage**: This error is stored in an error feedback buffer with exponential decay
3. **Error Compensation**: In the next step, the accumulated error is added to the current gradient before computing the update
4. **Iterative Refinement**: Over multiple steps, the error feedback mechanism helps recover from quantization losses

## Mathematical Formulation

For a parameter θ at step t:

1. **Intended Update**: `u_t = -lr * g_t`
2. **Actual Update**: `ũ_t = quantize(u_t)` (where quantize() may be identity for ErrorFeedbackMuon)
3. **Error**: `e_t = u_t - ũ_t`
4. **Error Feedback Buffer**: `b_{t+1} = decay * b_t + (1-decay) * e_t`
5. **Next Gradient**: `g_{t+1} = g_{t+1} + b_{t+1}`

## Benefits

- **Improved Convergence**: Error feedback helps maintain optimization quality even with approximations
- **Communication Efficiency**: Enables aggressive gradient compression in distributed training
- **Stability**: Reduces the impact of quantization noise on training dynamics
- **Flexibility**: Configurable decay rates allow tuning for different scenarios

## When to Use

- **ErrorFeedbackMuon**: When you want improved stability in standard training
- **ErrorFeedbackQuantizedMuon**: When implementing communication-efficient distributed training
- **Low-bit Training**: When using aggressive gradient quantization (4-8 bits)
- **Federated Learning**: When gradients are compressed before transmission

## Testing

A test script `test_error_feedback.py` is provided to demonstrate the optimizers:

```bash
python3 test_error_feedback.py
```

This script compares the performance of different optimizers on a simple optimization problem.

## Integration with Existing Code

The new optimizers can be easily integrated into the existing training loop by replacing the optimizer:

```python
# Replace this line:
# optimizer2 = NormalizedMuon(filter_params, lr=0.45, momentum=0.65, nesterov=True, sgd_coeff=0.6)

# With one of these:
optimizer2 = ErrorFeedbackMuon(filter_params, lr=0.45, momentum=0.65, nesterov=True, sgd_coeff=0.6, error_feedback_decay=0.9)
# or
optimizer2 = ErrorFeedbackQuantizedMuon(filter_params, lr=0.45, momentum=0.65, nesterov=True, sgd_coeff=0.6, error_feedback_decay=0.9, quantization_bits=8)
```

## References

- Error Feedback in Communication-Efficient Distributed Optimization
- Gradient Compression in Distributed Training
- Quantized Neural Network Training
