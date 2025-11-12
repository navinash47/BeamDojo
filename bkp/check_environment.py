#!/usr/bin/env python3
"""Script to check if the environment is properly configured for BeamDojo."""

import sys
import os

def check_conda_env():
    """Check if conda environment is activated."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env == "env_isaaclab":
        print(f"✓ Conda environment: {conda_env}")
        return True
    else:
        print(f"✗ Conda environment: {conda_env} (expected: env_isaaclab)")
        print("  Run: conda activate env_isaaclab")
        return False

def check_isaaclab_path():
    """Check if ISAACLAB_PATH is set."""
    isaaclab_path = os.environ.get("ISAACLAB_PATH", "")
    if isaaclab_path:
        if os.path.exists(isaaclab_path):
            print(f"✓ ISAACLAB_PATH: {isaaclab_path}")
            return True
        else:
            print(f"✗ ISAACLAB_PATH points to non-existent directory: {isaaclab_path}")
            return False
    else:
        print("✗ ISAACLAB_PATH not set")
        print("  Run: source setup_isaaclab.sh")
        return False

def check_imports():
    """Check if IsaacLab packages can be imported."""
    packages = ["isaaclab", "isaaclab_rl", "isaaclab_tasks"]
    all_ok = True
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ Import {package}: OK")
        except ImportError as e:
            print(f"✗ Import {package}: FAILED - {e}")
            all_ok = False
    
    return all_ok

def check_pythonpath():
    """Check if PYTHONPATH includes IsaacLab source."""
    pythonpath = os.environ.get("PYTHONPATH", "")
    isaaclab_path = os.environ.get("ISAACLAB_PATH", "")
    
    if not isaaclab_path:
        print(f"✗ ISAACLAB_PATH not set, cannot check PYTHONPATH")
        return False
    
    # Check if IsaacLab source is in PYTHONPATH
    if f"{isaaclab_path}/source" in pythonpath:
        print(f"✓ PYTHONPATH includes IsaacLab source")
        print(f"  PYTHONPATH: {pythonpath}")
        return True
    else:
        print(f"✗ PYTHONPATH does not include IsaacLab source")
        print(f"  Current PYTHONPATH: {pythonpath if pythonpath else '(empty)'}")
        print(f"  Expected to include: {isaaclab_path}/source")
        print("  Run: source setup_isaaclab.sh")
        return False

def check_cli_args():
    """Check if cli_args module can be found."""
    try:
        import cli_args
        print(f"✓ Import cli_args: OK")
        return True
    except ImportError:
        # Try to find it in scripts directory
        scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
        if os.path.exists(os.path.join(scripts_dir, "cli_args.py")):
            print(f"✓ cli_args.py found in scripts directory")
            print("  Note: Make sure to run from BeamDojo root directory")
            return True
        else:
            print(f"✗ cli_args module not found")
            return False

def main():
    """Run all checks."""
    print("=" * 60)
    print("BeamDojo Environment Check")
    print("=" * 60)
    print()
    
    results = []
    
    print("1. Conda Environment:")
    results.append(check_conda_env())
    print()
    
    print("2. Environment Variables:")
    results.append(check_isaaclab_path())
    results.append(check_pythonpath())
    print()
    
    print("3. Python Imports:")
    results.append(check_imports())
    print()
    
    print("4. CLI Args Module:")
    results.append(check_cli_args())
    print()
    
    print("=" * 60)
    if all(results):
        print("✓ All checks passed! Environment is properly configured.")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print()
        print("Quick fix:")
        print("  1. conda activate env_isaaclab")
        print("  2. source setup_isaaclab.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())

