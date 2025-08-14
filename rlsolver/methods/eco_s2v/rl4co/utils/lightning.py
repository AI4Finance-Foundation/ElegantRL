import os

import lightning as L
import torch
from omegaconf import DictConfig

# from rl4co.
from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def get_lightning_device(lit_module: L.LightningModule) -> torch.device:
    """Get the device of the Lightning module before setup is called
    See device setting issue in setup https://github.com/Lightning-AI/lightning/issues/2638
    """
    try:
        if lit_module.trainer.strategy.root_device != lit_module.device:
            return lit_module.trainer.strategy.root_device
        return lit_module.device
    except Exception:
        return lit_module.device


def remove_key(config, key="wandb"):
    """Remove keys containing 'key`"""
    new_config = {}
    for k, v in config.items():
        if key in k:
            continue
        else:
            new_config[k] = v
    return new_config


def clean_hydra_config(
        config, keep_value_only=True, remove_keys="wandb", clean_cfg_path=True
):
    """Clean hydra config by nesting dictionary and cleaning values"""
    # Remove keys containing `remove_keys`
    if not isinstance(remove_keys, list):
        remove_keys = [remove_keys]
    for key in remove_keys:
        config = remove_key(config, key=key)

    new_config = {}
    # Iterate over config dictionary
    for key, value in config.items():
        # If key contains slash, split it and create nested dictionary recursively
        if "/" in key:
            keys = key.split("/")
            d = new_config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value["value"] if keep_value_only else value
        else:
            new_config[key] = value["value"] if keep_value_only else value

    cfg = DictConfig(new_config)

    if clean_cfg_path:
        # Clean cfg_path recursively substituting root_dir with cwd
        root_dir = cfg.paths.root_dir

        def replace_dir_recursive(d, search, replace):
            for k, v in d.items():
                if isinstance(v, dict) or isinstance(v, DictConfig):
                    replace_dir_recursive(v, search, replace)
                elif isinstance(v, str):
                    if search in v:
                        d[k] = v.replace(search, replace)

        replace_dir_recursive(cfg, root_dir, os.getcwd())

    return cfg
