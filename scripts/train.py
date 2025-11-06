"""Training script using Hydra + Lightning."""

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
from pathlib import Path
from typing import Any

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from project_name.utils.logging_utils import setup_logger

log = setup_logger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiate callbacks from config."""
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping...")
        return callbacks

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiate loggers from config."""
    loggers: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return loggers

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> float | None:
    """Main training function.

    Args:
        cfg: Hydra config

    Returns:
        Optional metric value for hyperparameter optimization
    """
    # Set environment variable for paths
    if "PROJECT_ROOT" not in os.environ:
        project_root = Path(__file__).parent.parent
        os.environ["PROJECT_ROOT"] = str(project_root)
        log.info(f"Set PROJECT_ROOT to {project_root}")

    # Print config
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
        log.info(f"Set seed to {cfg.seed}")

    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Instantiate model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Instantiate callbacks
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Instantiate loggers
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    # Instantiate trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Log hyperparameters
    if loggers:
        log.info("Logging hyperparameters!")
        for logger in loggers:
            logger.log_hyperparams(
                {"model": cfg.model, "data": cfg.data, "trainer": cfg.trainer}
            )

    # Train
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    # Test
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = "best"
        if not cfg.get("train") or cfg.trainer.get("fast_dev_run"):
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Return metric for hyperparameter optimization
    metric_dict = trainer.callback_metrics
    metric_value = metric_dict.get("val/acc_best")

    if metric_value is not None:
        return float(metric_value)
    return None


if __name__ == "__main__":
    main()
