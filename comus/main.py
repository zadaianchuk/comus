import gc
import logging
import os
import shutil

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from comus.clustering import salient_objects_clustering, save_pseudolabels_saliency
from comus.semseg import eval_model, save_pseudolables_model, self_training
from comus.utils import update_dirs

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)


@hydra.main(version_base=None, config_path="../configs", config_name="config_comus")
def train_comus(cfg):
    """COMUS training pipeline. Clustering is done using 0 iteration.
    Next iterations self-training is done.

    Args:
    cfg: hydra config
    """

    dataset_results_dir = os.getcwd()
    for i in range(cfg.self_training.n_iter):
        output_dir, pseudolabels_dir, prev_output_dir = update_dirs(dataset_results_dir, i)
        if i == 0:
            labels, indexes = salient_objects_clustering(cfg, evaluate=False)
            save_pseudolabels_saliency(labels, indexes, pseudolabels_dir, cfg)
        else:
            checkpoint_path = os.path.join(
                prev_output_dir, f"checkpoint{cfg.self_training.n_epochs:04}.pth"
            )
            if cfg.dataset.name == "pascal":
                # update data used and number of epochs
                cfg.eval_model.pseudolables_split = cfg.self_training.second_stage.train_split
                cfg.self_training.train_split = cfg.self_training.second_stage.train_split
                cfg.self_training.n_epochs = cfg.self_training.second_stage.n_epochs
            save_pseudolables_model(pseudolabels_dir, checkpoint_path, cfg)
        torch.cuda.empty_cache()
        gc.collect()
        log.info("Start iteration")
        mp.spawn(
            self_training,
            args=(output_dir, pseudolabels_dir, cfg),
            nprocs=8,
            join=True,
        )
        checkpoint_path = os.path.join(output_dir, f"checkpoint{cfg.self_training.n_epochs:04}.pth")
        log.info("Stop iteration")
        eval_model(checkpoint_path, output_dir, cfg)
    # clean generated pseudolables
    if os.path.exists(pseudolabels_dir):
        shutil.rmtree(pseudolabels_dir)


if __name__ == "__main__":
    train_comus()
