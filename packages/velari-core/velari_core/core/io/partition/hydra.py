import hydra
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


def _register_root_resolver() -> None:
    if not OmegaConf.has_resolver("root"):
        from ...utils.env_utils import read_root_dir

        OmegaConf.register_new_resolver("root", lambda rel: str(Path(read_root_dir()) / rel))


def init_hydra() -> None:
    """Clear the global Hydra instance if already initialised, preparing it for a fresh compose call."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def read_hydra_defaults(config_dir: str, config_name: str) -> DictConfig:
    """Load Hydra configuration from a directory, registering the tuple resolver if absent.

    Changes the working directory temporarily to the script's location before
    initialising Hydra, then restores the original directory on exit regardless
    of whether the composition succeeds or raises.

    Args:
        config_dir: Absolute path to the directory containing the Hydra config files.
        config_name: Name of the top-level config file, without the ``.yaml`` extension.

    Returns:
        The composed DictConfig produced by Hydra, with all overrides and defaults applied.

    Examples:
        >>> from pathlib import Path
        >>> cfg = read_hydra_defaults(config_dir="config", config_name="config")

        Retrieve the logging section and pass it downstream:

        >>> from velari_core.core.utils.env_utils import read_root_dir
        >>> config_dir = Path(read_root_dir()) / "config"
        >>> cfg = read_hydra_defaults(str(config_dir), "config")
        >>> log_filename = cfg.logging  # resolve the logging config path
    """

    def resolve_tuple(*args):
        return tuple(args)

    if not OmegaConf.has_resolver("as_tuple"):
        OmegaConf.register_new_resolver("as_tuple", resolve_tuple)

    # Save the current working directory
    old_cwd = os.getcwd()
    script_dir = Path(__file__).parent.resolve()
    rel_config_dir = os.path.relpath(config_dir, script_dir)
    try:
        os.chdir(script_dir)
        with hydra.initialize(config_path=rel_config_dir, version_base=None):
            cfg = hydra.compose(config_name=config_name)
            return cfg
    finally:
        # Restore the original working directory
        os.chdir(old_cwd)


def read_hydra_compose(
    config_path: str,
    config_name: str = "config.yaml",
    overrides: Optional[List[str]] = None,
) -> Tuple[DictConfig, str]:
    """Compose Hydra configuration from a given directory and config file.

    Args:
        config_path: Path to the directory containing the Hydra config files.
        config_name: Name of the config file to compose. Defaults to "config.yaml".
        overrides: Optional list of Hydra override strings (e.g. searchpath additions).

    Returns:
        A tuple containing the composed DictConfig and its YAML string representation.

    Examples:
        >>> from velari_core.core.utils.env_utils import read_root_dir
        >>> from pathlib import Path
        >>> config_path = str(Path(read_root_dir()) / "config")
        >>> cfg, cfg_yaml = read_hydra_compose(config_path, "config.yaml")
        >>> print(cfg_yaml)
    """
    _register_root_resolver()
    init_hydra()
    with hydra.initialize_config_dir(
        config_dir=str(Path(config_path).resolve()),
        version_base=None,
    ):
        config = hydra.compose(config_name=config_name, overrides=overrides or [])
        config_yaml = OmegaConf.to_yaml(config)
        return config, config_yaml


def read_hydra(filepath: str, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Optional[DictConfig]:
    """Load an OmegaConf configuration directly from a YAML file.

    Args:
        filepath: Path to the YAML configuration file to load.
        *args: Unused positional arguments reserved for future extension.
        **kwargs: Unused keyword arguments reserved for future extension.

    Returns:
        A DictConfig parsed from the given YAML file, or None if the file is empty
        or cannot be resolved by OmegaConf.

    Examples:
        >>> from pathlib import Path
        >>> cfg = read_hydra(Path("config") / "experimentation" / "experiment.yaml")

        Load an experiment template then inject a run name before use:

        >>> from velari_core.core.utils.env_utils import read_root_dir
        >>> yaml_path = Path(read_root_dir()) / "config" / "experimentation" / "experiment.yaml"
        >>> cfg = read_hydra(yaml_path)
        >>> cfg.experiment.name = "bert-finetune-v1"
    """
    _register_root_resolver()
    cfg = OmegaConf.load(filepath)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected a DictConfig from {filepath}, got {type(cfg).__name__}")
    return cfg
