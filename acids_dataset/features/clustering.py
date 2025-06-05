import torch, numpy as np
from absl import logging
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

def hash_from_clustering(dataset, feature, n_centroids, pca_target_dim=None, pca_whiten=False, random=True, n_examples=None, write=False, verbose=False, **kwargs):
    assert feature in dataset.features, "feature %s not found in feature_hash"%(feature)
    # apply clustering
    if n_examples is None: n_examples = len(dataset)
    if verbose: logging.info("clustering feature %s on %d exemples..."%(feature, n_examples))
    if random: 
        idx = range(n_examples)
    else:
        idx = torch.randperm(len(dataset))[:n_examples].tolist()
    features = []
    for i in idx:
        data = dataset.get(i, output_pattern=feature).flatten()
        if isinstance(data, torch.Tensor): data = data.numpy()
        features.append(data)

    # optional PCA pass for high-dim input
    if pca_target_dim: 
        if verbose: logging.info("computing PCA...")
        assert pca_target_dim>0, "pca_target_dim must be positive"
        pca_obj = PCA(n_components=min(pca_target_dim, n_examples), whiten=pca_whiten, random_state=0)
        features = pca_obj.fit_transform(features)
    
    features = np.stack(features, axis=0)
    kmeans = MiniBatchKMeans(n_clusters=n_centroids, random_state=0, verbose=verbose, **kwargs)
    kmeans.fit(features)

    if verbose: logging.info("computing clusters for all dataset...")
    feature_hash = {i: [] for i in range(n_centroids)}
    for i in range(len(dataset)):
        data = dataset.get(i, output_pattern=feature).flatten()
        if isinstance(data, torch.Tensor): data = data.numpy()
        k = kmeans.predict(data[None])
        feature_hash[int(k)].append(dataset.keys[i])

    if write: 
        dataset.loader.writer.add_feature_hash(dataset.path, feature, feature_hash)
        dataset.loader.writer.append_to_feature_metadata(dataset.path, feature, {'cluster_for_hash': kmeans})

    return kmeans
        

