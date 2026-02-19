"""
Main orchestrator for LtM experiments.
Handles inference-only prompt tuning tasks.
"""
import sys
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main orchestrator - invokes inference.py as subprocess.
    This is an inference-only task, so we don't call train.py.
    """
    
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: ConfigAttributeError: Key 'method' is not in struct
    # [CAUSE]: Run config is loaded under 'run' namespace via Hydra defaults,
    #          but code tries to access cfg.method instead of cfg.run.method
    # [FIX]: Changed cfg.method to cfg.run.method to match Hydra config structure
    #
    # [OLD CODE]:
    # print(f"Starting run: {cfg.run.run_id}")
    # print(f"Mode: {cfg.mode}")
    # print(f"Method: {cfg.method.type}")
    #
    # [NEW CODE]:
    print(f"Starting run: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Method: {cfg.run.method.type}")
    
    # For this inference-only task, invoke inference.py
    cmd = [
        sys.executable, "-u", "-m", "src.inference",
        f"run={cfg.run.run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, check=True)
    
    print(f"Completed run: {cfg.run.run_id}")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
