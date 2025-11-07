"""Evaluation script using Hydra + Lightning."""

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
from pathlib import Path

import hydra
import lightning as L
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf

from modern_vit.utils.logging_utils import setup_logger
from modern_vit.utils.parameter_utils import print_model_parameters

log = setup_logger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Main evaluation function.

    Args:
        cfg: Hydra config
    """
    # Set environment variable for paths
    if "PROJECT_ROOT" not in os.environ:
        project_root = Path(__file__).parent.parent
        os.environ["PROJECT_ROOT"] = str(project_root)
        log.info(f"Set PROJECT_ROOT to {project_root}")

    # Print config
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Instantiate model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Print model parameters
    print_model_parameters(model)

    # Instantiate trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    # Check checkpoint path
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        log.info(f"Will load checkpoint from: {ckpt_path}")
    else:
        log.warning("No checkpoint path specified. Model will use randomly initialized weights.")

    # Evaluate
    log.info("Starting evaluation!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
