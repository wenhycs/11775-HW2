import numpy as np
import pickle
from pyturbo import Stage


class BagOfWords(Stage):

    """
    Input: features [N x D]
    Output: bag-of-words [W]
    """

    def allocate_resource(self, resources, *, weight_path):
        self.weight_path = weight_path
        self.weight = None
        return [resources]

    def reset(self):
        if self.weight is None:
            with open(self.weight_path, 'rb') as f:
                self.weight = pickle.load(f)

    def get_bag_of_words(self, features: np.ndarray) -> np.ndarray:
        """
        features: [N x D]

        Return: count of each word, [W]
        """
        # TODO: Generate bag of words
        # Calculate pairwise distance between each feature and each cluster,
        # assign each feature to the nearest cluster, and count
        raise NotImplementedError

    def get_video_feature(self, bags: np.ndarray) -> np.ndarray:
        """
        bags: [B x W]

        Return: pooled vector, [W]
        """
        # TODO: Aggregate frame-level bags into a video-level feature.
        raise NotImplementedError

    def process(self, task):
        features = task.content
        bags = []
        for frame_features in features:
            bag = self.get_bag_of_words(frame_features)
            assert isinstance(bag, np.ndarray)
            assert bag.shape == self.weight.shape[:1]
            bags.append(bag)
        bags = np.stack(bags)
        video_bag = self.get_video_feature(bags)
        assert isinstance(video_bag, np.ndarray)
        assert video_bag.shape == self.weight.shape[:1]
        return task.finish(video_bag)
