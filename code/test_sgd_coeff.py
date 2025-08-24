"""
Simple test script to verify the setup works before running the full experiment.
dull code
"""

import torch
from airbench_muon import CifarNet, NormalizedMuon

def test_sgd_coeff():
    """Test that we can create a model and optimizer with different sgd_coeff values."""
    
    print("Testing SGD coefficient setup...")
    
    # Create a simple model
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    
    # Test different sgd_coeff values
    test_coeffs = [-0.1, 0.0, 0.5, 1.0, 1.1]
    
    for coeff in test_coeffs:
        try:
            # Get filter parameters
            filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
            
            # Create optimizer with this coefficient
            optimizer = NormalizedMuon(filter_params, lr=0.4, momentum=0.65, nesterov=True, sgd_coeff=coeff)
            
            print(f"  ✓ Successfully created NormalizedMuon with sgd_coeff = {coeff}")
            
        except Exception as e:
            print(f"  ✗ Failed to create NormalizedMuon with sgd_coeff = {coeff}: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_sgd_coeff()
