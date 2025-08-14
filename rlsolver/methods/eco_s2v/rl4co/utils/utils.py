import importlib
import platform
import sys
import warnings
from importlib.util import find_spec
from typing import Callable

import hydra
from lightning import Callback
from lightning.pytorch.loggers.logger import Logger
# Import the necessary PyTorch Lightning component
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _AcceleratorConnector,
)
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from rlsolver.methods.eco_s2v.rl4co.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished or failed
    - Logging the exception if occurs
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # close loggers (even if exception occurs so multirun won't fail)
            close_loggers()

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config."""
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiates loggers from config."""
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    ## Note: we do not use the data config, since it is dealt with in the model
    ## which is a `LightningModule`
    # hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def merge_with_defaults(_config=None, **defaults) -> dict:
    """Merge configuration with default values.

    This function merges a provided configuration dictionary with default values.
    If no configuration is provided (`_config` is None), it returns the default values.
    If a dictionary is provided, it updates the defaults dictionary with the values from the provided dictionary.
    Otherwise, it sets all keys in the defaults dictionary to `_config`.

    Args:
        _config: Configuration to merge. Defaults to None.
        **defaults: Default values to merge with the configuration.

    Returns:
        dict: Merged configuration with default values.
    """
    if _config is None:
        return defaults
    elif isinstance(_config, (DictConfig, dict)):
        defaults.update(dict(**_config))  # type: ignore
        return defaults
    else:
        return {key: _config for key in defaults.keys()}


def show_versions():
    """
    This function prints version information that is useful when filing bug
    reports. Inspired by https://github.com/PyVRP/PyVRP
    """

    modules = {
        "rl4co": "rl4co",
        "torch": "torch",
        "lightning": "pytorch_lightning",  # Updated module name if necessary
        "torchrl": "torchrl",
        "tensordict": "tensordict",
        "numpy": "numpy",
        "pytorch_geometric": "torch_geometric",
        "hydra-core": "hydra",
        "omegaconf": "omegaconf",
        "matplotlib": "matplotlib",
    }

    # Find the longest module name for formatting
    longest_name = max(len(name) for name in modules.keys())

    print("INSTALLED VERSIONS")
    print("-" * (longest_name + 20))
    # modules
    for name, module in modules.items():
        try:
            imported_module = importlib.import_module(module)
            version = imported_module.__version__
        except ImportError:
            version = "Not installed"
        print(f"{name.rjust(longest_name)} : {version}")
    # platform information
    print(f'{"Python".rjust(longest_name)} : {sys.version.split()[0]}')
    print(f'{"Platform".rjust(longest_name)} : {platform.platform()}')
    try:
        lightning_auto_device = _AcceleratorConnector()._choose_auto_accelerator(None)
    except Exception:
        lightning_auto_device = _AcceleratorConnector()._choose_auto_accelerator()
    # lightning hardware accelerators
    print(f'{"Lightning device".rjust(longest_name)} : {lightning_auto_device}')
