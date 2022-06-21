import logging
import os

import hydra
from omegaconf import OmegaConf

from comus.semseg import eval_model

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)


@hydra.main(version_base=None, config_path="../configs", config_name="config_comus_eval")
def eval_comus(cfg):
    """COMUS training pipeline. Clustering is done using 0 iteration.

    Next iterations self-training is done.

    Args:
    cfg: hydra config
    """
    output_dir = cfg.output_dir
    checkpoint_path = os.path.join(output_dir, f"checkpoint{cfg.checkpoint:04}.pth")
    eval_output_dir = os.path.join(cfg.output_dir, cfg.name)
    eval_model(checkpoint_path, eval_output_dir, cfg)


if __name__ == "__main__":
    eval_comus()
