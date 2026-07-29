"""Tests for Hydra configuration loading."""
from pathlib import Path
from omegaconf import OmegaConf


CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def test_config_file_exists():
    assert CONFIG_PATH.exists(), f"config.yaml not found at {CONFIG_PATH}"

def test_config_loads():
    cfg = OmegaConf.load(CONFIG_PATH)
    assert cfg is not None

def test_config_logging_section():
    cfg = OmegaConf.load(CONFIG_PATH)
    assert "logging" in cfg
