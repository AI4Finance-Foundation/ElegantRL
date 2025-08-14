import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from rlsolver.methods.eco_s2v.rl4co.utils import pylogger

log = pylogger.get_pylogger(__name__)


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


def instantiate_loggers(logger_cfg: DictConfig, model) -> list[Logger]:
    """Instantiates loggers from config."""

    logger_list: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger_list

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            if hasattr(lg_conf, "log_gradients"):
                log_gradients = lg_conf.get("log_gradients", False)
                # manually remove parameter, since pop doesnt work on DictConfig
                del lg_conf.log_gradients
            else:
                log_gradients = False
            logger = hydra.utils.instantiate(lg_conf)
            if hasattr(logger, "watch") and log_gradients:
                # make use of wandb gradient statistics logger
                logger.watch(model, log_graph=False)
            logger_list.append(logger)

    return logger_list
