"""Configuration loading utilities."""
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def merge_config(base: Dict, overrides: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, val in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_config(result[key], val)
        else:
            result[key] = val
    return result
