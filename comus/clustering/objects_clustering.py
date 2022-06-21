import logging
import os

import hydra
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from comus.clustering.cluster import cluster
from comus.clustering.evaluate import evaluate_clustering
from comus.clustering.load import get_dataset_loader, load_model
from thirdparty.utils import mkdir_if_missing

log = logging.getLogger(__name__)


def salient_objects_clustering(cfg, evaluate=True):
    """COMUS clustering pipeline.

    Args:
        cfg: hydra config.
        pseudolabels_dir (str, optional): dir path to store pseudolables from clustering.
        evaluate (bool, optional): if False only pseudolables are computed.
    """
    dataset_results_dir = os.getcwd()
    mkdir_if_missing(dataset_results_dir)
    data_path = os.path.join(dataset_results_dir, "precomputed_features.npz")
    model = load_model(model_name=cfg.features.model, data_parallel=cfg.dataset.data_parallel)
    loader, _, _ = get_dataset_loader(cfg=cfg)
    features, indexes = collect_features(loader, model, cfg, data_path)
    labels, is_core = cluster(features, cfg.clustering)
    if is_core is not None:
        labels, indexes = labels[is_core], indexes[is_core]
    log.info(
        f"Overall {labels.shape[0]} features were clustered to {cfg.dataset.n_classes} clusters."
    )
    if evaluate:
        evaluate_clustering(dataset_results_dir, labels, indexes, cfg)
    return labels, indexes


def collect_features(loader, model, config, data_path=None):

    if config.features.use_saved and os.path.isfile(data_path):
        log.info(
            """Using cashed features.
            Use with caution if some changes where made to the input features. \n
            Use use_saved=False, to turn off features cashing."""
        )
        data = np.load(data_path)
        features = data["features"]
        indexes = data["indexes"]
    else:
        all_features = []
        all_indexes = []
        for indexes, images, _ in tqdm(loader):
            features = model(images.cuda()).cpu().detach().numpy()
            all_features.append(features)
            all_indexes.append(indexes)
        features = np.concatenate(all_features)
        indexes = np.concatenate(all_indexes)
        if data_path is not None:
            np.savez(data_path, indexes=indexes, features=features)
    return features, indexes


@hydra.main(version_base=None, config_path="../../configs", config_name="config_clustering")
def run_segment_clustering(cfg):
    """Run clustering independently.

    Args:
        cfg : hydra config
    """
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
    return salient_objects_clustering(cfg, evaluate=True)


if __name__ == "__main__":
    run_segment_clustering()
