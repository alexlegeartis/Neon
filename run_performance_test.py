#!/usr/bin/env python3
"""
Simple runner script for the performance test.
This ensures the test runs with Python 3 and provides a convenient entry point.
"""

import subprocess
import sys
import os

def main():
    """Run the performance test script."""
    script_path = os.path.join("code", "performance_test_256x2304.py")
    
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found!")
        sys.exit(1)
    
    print("Running performance test for 256x2304 matrices...")
    print("Comparing zeropower_via_newtonschulz5 vs one_sv_svds_approximation")
    print("-" * 60)
    
    try:
        # Run the test script with Python 3
        result = subprocess.run([sys.executable, script_path], 
                              check=True, 
                              capture_output=False)
        print("\nTest completed successfully!")
        print(f"Results saved in: {os.path.join('code', 'performance_comparison_256x2304.png')}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running test: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
