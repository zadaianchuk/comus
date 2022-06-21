import numpy as np
from scipy.spatial.distance import cosine
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.cluster._kmeans import k_means
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import kneighbors_graph
from sklearn_extra.cluster import KMedoids


class KMedoidsWrapper(KMedoids):
    def __init__(
        self,
        n_clusters=8,
        metric="euclidean",
        method="alternate",
        init="heuristic",
        n_init=10,
        max_iter=300,
        random_state=None,
    ):
        self._n_init = n_init
        super().__init__(n_clusters, metric, method, init, max_iter, random_state)

    def fit(self, X, y=None):
        best_inertia = None
        for _ in range(self._n_init):
            # run a k-means once
            super().fit(X)
            labels, centers, inertia = (
                self.labels_,
                self.cluster_centers_,
                self.inertia_,
            )

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels
                best_inertia = inertia
                best_centers = centers

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self


def spectral_clustering(
    X,
    n_clusters=20,
    n_components=20,
    n_neighbors=30,
    eigen_solver="amg",
    eigen_tol=0.0,
    n_init=10,
    random_state=None,
    n_jobs=10,
):
    """NN Spectral Clustering similar to original sklearn with additional return of the eigenmaps."""
    n_components = n_clusters if n_components is None else n_components

    connectivity = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=True, n_jobs=n_jobs)
    affinity_matrix = 0.5 * (connectivity + connectivity.T)

    maps = spectral_embedding(
        affinity_matrix,
        n_components=n_components,
        eigen_solver=eigen_solver,
        random_state=random_state,
        eigen_tol=eigen_tol,
        drop_first=False,
    )
    centers, labels, _ = k_means(maps, n_clusters, random_state=random_state, n_init=n_init)
    return labels, centers, maps


class SpectralClustering(BaseEstimator):
    """SpectralClustering estimator with NN affinity and kmeans.

    Additionally to cluster labels, stores spectral embeddings
    and kmeans centers.
    """

    def __init__(
        self,
        n_clusters,
        n_components,
        n_neighbors=30,
        eigen_solver="amg",
        eigen_tol=0.0,
        n_init=10,
        n_jobs=20,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.n_neighbor = n_neighbors
        self.eigen_tol = eigen_tol
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X):
        """Fit data features X to spectral_clustering.

        In additional to assigned lables, stores
        and cluster_centers proccessed features.
        """
        self.labels_, self.cluster_centers_, self.maps_ = spectral_clustering(
            X=X,
            n_clusters=self.n_clusters,
            n_components=self.n_components,
            eigen_solver=self.eigen_solver,
            eigen_tol=self.eigen_tol,
            n_init=self.n_init,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        return self


clusterings = {
    "spectral_clustering": SpectralClustering,
    "kmedoids": KMedoidsWrapper,
    "agglomerative_clustering": AgglomerativeClustering,
    "kmeans": KMeans,
    "minibatch_kmeans": MiniBatchKMeans,
}


def cluster(features, clustering_config):

    clustering = clusterings[clustering_config.name](**clustering_config.params)
    clustering = clustering.fit(features)

    # add is_core attribute

    if clustering_config.name != "spectral_clustering":
        maps = features
    else:
        maps = clustering.maps_
    clustering.core_size = clustering_config.core_size
    assert clustering_config.core_size >= 0 and clustering_config.core_size <= 100
    if clustering_config.name == "agglomerative_clustering":
        assert clustering_config.core_size == 100
        is_core = np.ones_like(clustering.labels_).astype(bool)
    else:
        clustering.dist_ = find_dist(maps, clustering.cluster_centers_, clustering.labels_)
        is_core = get_core_samples(
            clustering.dist_,
            clustering.labels_,
            percentage=clustering_config.core_size,
        )

    return clustering.labels_, is_core


def get_core_samples(dist_to_center, labels, percentage=80):
    if percentage != 100:
        is_core = np.zeros_like(dist_to_center)
        unique_lables = set(labels)
        th_dict = {
            u_l: np.percentile(dist_to_center[labels == u_l], percentage) for u_l in unique_lables
        }
        for u_l, th in th_dict.items():
            mask = np.logical_and(dist_to_center < th, labels == u_l)
            is_core[mask] = 1.0
    else:
        is_core = np.ones_like(dist_to_center)
    return is_core.astype(bool)


def find_dist(features, centers, labels):
    dist = np.array([[cosine(feature, center) for center in centers] for feature in features])
    return np.array([dist[i, l] for i, l in enumerate(labels)])
