import pyrootutils

root = str(pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "README.md"],
    pythonpath=True,
    dotenv=True))

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "README.md" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

# Hack importing pandas here to bypass some conflicts with hydra
import pandas as pd

from typing import List, Tuple

import hydra
import torch
import torch_geometric
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from plyfile import PlyData, PlyElement
import numpy as np
import os
from src import utils

# Registering the "eval" resolver allows for advanced config
# interpolation with arithmetic operations:
# https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html
if not OmegaConf.has_resolver('eval'):
    OmegaConf.register_new_resolver('eval', eval)

log = utils.get_pylogger(__name__)

def save_classified_ply(data: torch_geometric.data.Data, pred_labels: torch.Tensor, output_dir: str):
    """
    Saves a classified PLY file with predicted labels.

    Args:
        data (torch_geometric.data.Data): The input data object containing point cloud data.
        pred_labels (torch.Tensor): Predicted labels for each point.
        output_dir (str): Directory to save the classified PLY file.
    """
    original_ply_path = data.file_path
    output_ply_path = get_output_ply_path(original_ply_path, output_dir)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)

    # Read the original PLY file
    plydata = PlyData.read(original_ply_path)
    vertex_data = plydata['vertex'].data

    # Verify the number of points matches
    num_vertices = len(vertex_data)
    if len(pred_labels) != num_vertices:
        log.error(f"Number of predictions ({len(pred_labels)}) does not match number of vertices ({num_vertices}) in '{original_ply_path}'.")
        return

    # Define new data type with an additional 'pred' property
    # Assuming pred_labels are integers and num_classes <= 255
    new_dtype = vertex_data.dtype.descr + [('pred', 'u1')]
    new_vertex_data = np.empty(num_vertices, dtype=new_dtype)

    # Copy existing data
    for name in vertex_data.dtype.names:
        new_vertex_data[name] = vertex_data[name]

    # Add predictions
    new_vertex_data['pred'] = pred_labels.numpy().astype(np.uint8)

    # Create a new PlyElement
    new_vertex_element = PlyElement.describe(new_vertex_data, 'vertex')

    # Determine original file format (binary or ASCII)
    is_binary = plydata.header.get('format', '').startswith('binary')

    # Create a new PlyData object
    new_plydata = PlyData([new_vertex_element], text=not is_binary)

    # Write to the output PLY file
    try:
        new_plydata.write(output_ply_path)
        log.info(f"Saved classified PLY to '{output_ply_path}'.")
    except Exception as e:
        log.error(f"Failed to write classified PLY '{output_ply_path}': {e}")

def get_output_ply_path(original_ply_path: str, output_dir: str) -> str:
    """
    Generates the output PLY file path based on the original PLY file path.

    Args:
        original_ply_path (str): Path to the original PLY file.
        output_dir (str): Directory to save the classified PLY file.

    Returns:
        str: Path to the output classified PLY file.
    """
    filename = os.path.basename(original_ply_path)
    classified_filename = f"classified_{filename}"
    return os.path.join(output_dir, classified_filename)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    if float('.'.join(torch.__version__.split('.')[:2])) >= 2.0:
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)


    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)
    
    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model, dynamic=True)

    # log.info("Starting testing!")
    # trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    log.info("Starting testing!")
    predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    
    log.info("Processing and saving predictions...")
    output_dir = cfg.get("output_dir", "./classified_ply")
    for pred in predictions:
        data = pred[0]  # Assuming predict_step returns (batch, output)
        output = pred[1]
        
        # Extract predicted labels from output
        if hasattr(output, 'logits'):
            pred_labels = torch.argmax(output.logits, dim=1)
        elif isinstance(output, torch.Tensor):
            pred_labels = torch.argmax(output, dim=1)
        else:
            log.error(f"Unexpected output format: {type(output)}")
            continue

        save_classified_ply(data, pred_labels, output_dir=output_dir)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root + "/configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
