"""
Configuration utilities.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_previous_experiment_results(
    experiment_name: str,
    result_file: str,
    project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Load results from a previous experiment.
    
    Args:
        experiment_name: e.g., "exp1_architecture"
        result_file: e.g., "best_architecture.json"
        project_root: Project root path (auto-detected if None)
    
    Returns:
        Dictionary with results
    """
    if project_root is None:
        # Try to find project root
        current = Path.cwd()
        while current != current.parent:
            if (current / "shared").exists():
                project_root = current
                break
            current = current.parent
        else:
            raise FileNotFoundError("Could not find project root")
    
    exp_dir = project_root / experiment_name / "results"
    
    if not exp_dir.exists():
        raise FileNotFoundError(f"No results directory for {experiment_name}")
    
    # Find latest results folder (by timestamp)
    result_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
    
    if not result_dirs:
        raise FileNotFoundError(f"No results found in {exp_dir}")
    
    result_path = result_dirs[0] / result_file
    
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")
    
    with open(result_path, 'r') as f:
        return json.load(f)



