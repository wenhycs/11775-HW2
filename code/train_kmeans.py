import argparse
import functools
import os
import os.path as osp
import pickle
from typing import List

import numpy as np
import pandas as pd
from pyturbo import process_map
from sklearn.cluster import KMeans

from stages import LoadFeature


def select_features(features: List[np.ndarray]) -> np.ndarray:
    """
    features: list of [N x D]

    Return: selected features, [n x D]
    """
    # TODO: select subset of features for clustering
    raise NotImplementedError


def worker(video_id, *, args):
    feature_path = osp.join(args.feature_dir, f'{video_id}.pkl')
    features = LoadFeature.load_features(feature_path)
    selected_features = select_features(features)
    assert selected_features.ndim == 2
    assert selected_features.shape[1] == features[0].shape[1]
    return selected_features


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument('list_file_path')
    parser.add_argument('feature_dir')
    parser.add_argument('num_clusters', type=int)
    parser.add_argument('model_name')
    parser.add_argument(
        '--model_dir', default=osp.join(
            osp.dirname(__file__), '../data/kmeans'))
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)
    return args


def main(args):
    df = pd.read_csv(args.list_file_path)
    video_ids = df['Id']
    worker_fn = functools.partial(worker, args=args)
    map_fn = process_map if not args.debug else map
    video_features = np.concatenate([*map_fn(worker_fn, video_ids)])
    kmeans = KMeans(args.num_clusters, random_state=args.seed, verbose=1)
    kmeans.fit(video_features)
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = osp.join(args.model_dir, f'{args.model_name}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans.cluster_centers_, f)


if __name__ == '__main__':
    main(parse_args())
