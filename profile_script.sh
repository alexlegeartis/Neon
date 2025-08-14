#!/bin/bash

# Add local bin to PATH if not already there
export PATH="$HOME/.local/bin:$PATH"

# Run py-spy profiling
py-spy record -o profile.svg -- python3 code/muon_vs_neon.py
