# SGD Coefficient Experiment for Airbench Muon

This directory contains scripts to experiment with different `sgd_coeff` values in the NormalizedMuon optimizer and visualize the results.

## Files

- `airbench_muon.py` - Original training script with the Muon optimizer
- `run_sgd_coeff_experiment.py` - Main experiment script that tests different sgd_coeff values
- `test_sgd_coeff.py` - Simple test script to verify the setup works
- `requirements.txt` - Required Python packages

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have a CUDA-capable GPU and PyTorch with CUDA support.

## Usage

### 1. Test the Setup (Recommended First Step)

Before running the full experiment, test that everything works:

```bash
python test_sgd_coeff.py
```

This will verify that you can create models and optimizers with different `sgd_coeff` values.

### 2. Run the Full Experiment

Run the main experiment script to test different `sgd_coeff` values from -0.1 to 1.1:

```bash
python run_sgd_coeff_experiment.py
```

This will:
- Test 13 different `sgd_coeff` values: [-0.1, 0.0, 0.1, 0.2, ..., 1.0, 1.1]
- Run 20 training runs for each coefficient value
- Generate a bar chart showing the results
- Save the results to `sgd_coeff_experiment_results.pt`
- Save the plot as `sgd_coeff_results.png`

### 3. Expected Results

The script will test how the `sgd_coeff` parameter affects the model's accuracy:

- **sgd_coeff = 0**: Pure Muon optimizer (whitened updates)
- **sgd_coeff = 1**: Pure SGD optimizer (normalized gradients)
- **Values in between**: Linear interpolation between the two approaches

## Understanding the Results

The `sgd_coeff` parameter controls the balance between:
- **Muon-style updates**: Uses the `zeropower_via_newtonschulz5` function to whiten gradients
- **SGD-style updates**: Uses simple normalized gradients

- **Negative values**: Emphasize Muon-style updates even more
- **0**: Pure Muon optimizer
- **0.5**: Equal mix of both approaches
- **1**: Pure SGD optimizer
- **Values > 1**: Emphasize SGD-style updates even more

## Output Files

- `sgd_coeff_results.png` - Bar chart visualization
- `sgd_coeff_experiment_results.pt` - PyTorch tensor with all numerical results

## Performance Notes

- Each coefficient value requires ~20 training runs for statistical significance
- Total runtime depends on your GPU but expect several hours for the full experiment
- The script uses the same model architecture and training procedure as the original airbench_muon.py
- Results are averaged across multiple runs to account for training randomness

## Troubleshooting

- **CUDA out of memory**: Reduce `num_runs` in the experiment script
- **Import errors**: Make sure you're in the same directory as `airbench_muon.py`
- **Slow performance**: The script uses PyTorch compilation which may take time on first run