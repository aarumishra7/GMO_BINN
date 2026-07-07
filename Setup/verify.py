#!/usr/bin/env python
"""
Verification script: Check that all dependencies are installed and working.
Run this after setup.sh or setup.bat to confirm the environment is ready.

Usage:
    python verify.py
"""

import sys
import importlib

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

def check_module(name, display_name=None):
    """Check if a module is installed and print the version."""
    display_name = display_name or name
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{GREEN}✓{RESET} {display_name:20s} {version}")
        return True
    except ImportError as e:
        print(f"{RED}✗{RESET} {display_name:20s} NOT FOUND")
        print(f"   Error: {e}")
        return False

def main():
    print(f"\n{BLUE}{'='*50}{RESET}")
    print(f"{BLUE}GMO-BINN Environment Verification{RESET}")
    print(f"{BLUE}{'='*50}{RESET}\n")
    
    print("Checking core Python packages:\n")
    
    modules = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("yaml", "PyYAML"),
        ("torch", "PyTorch"),
        ("torchdiffeq", "torchdiffeq"),
    ]
    
    results = []
    for module_name, display_name in modules:
        results.append(check_module(module_name, display_name))
    
    print(f"\n{BLUE}Checking PyTorch details:{RESET}\n")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  (Running on CPU, which is fine for this project)")
    except Exception as e:
        print(f"  Error getting PyTorch details: {e}")
    
    print(f"\n{BLUE}Checking project files:{RESET}\n")
    
    import os
    files = [
        "config.yaml",
        "requirements.txt",
        "README.md",
        "src/grn_simulator.py",
        "src/models.py",
        "src/train.py",
        "src/evaluate.py",
        "src/utils.py",
    ]
    
    for filename in files:
        if os.path.exists(filename):
            print(f"{GREEN}✓{RESET} {filename}")
        else:
            print(f"{RED}✗{RESET} {filename} NOT FOUND")
            results.append(False)
    
    print(f"\n{BLUE}Checking config loading:{RESET}\n")
    
    try:
        import yaml
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        n_genes = cfg["grn"]["n_total"]
        n_conditions = cfg["data"]["n_conditions"]
        n_epochs = cfg["training"]["n_epochs"]
        print(f"{GREEN}✓{RESET} Config loaded successfully")
        print(f"  • {n_genes} genes configured")
        print(f"  • {n_conditions} GMO conditions")
        print(f"  • {n_epochs} training epochs")
    except Exception as e:
        print(f"{RED}✗{RESET} Config loading failed: {e}")
        results.append(False)
    
    # Summary
    print(f"\n{BLUE}{'='*50}{RESET}")
    if all(results):
        print(f"{GREEN}All checks passed! You're ready to go.{RESET}")
        print(f"\nNext steps:")
        print(f"  1. Review config.yaml for hyperparameters")
        print(f"  2. Run Phase 1: python -m src.grn_simulator")
        print(f"  3. Run full pipeline: python run_all.py")
        print(f"{BLUE}{'='*50}{RESET}\n")
        return 0
    else:
        print(f"{RED}Some checks failed. See above for details.{RESET}")
        print(f"\nTroubleshooting:")
        print(f"  • Make sure the virtual environment is activated")
        print(f"  • Try: pip install -r requirements.txt --no-cache-dir")
        print(f"  • Check your internet connection during pip install")
        print(f"{BLUE}{'='*50}{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())