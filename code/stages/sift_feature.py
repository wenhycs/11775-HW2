import cv2
import numpy as np
import torch

from pyturbo import Stage, Task


class SIFTFeature(Stage):

    """
    Input: batch of frames [B x H x W x C]
    Output: yield SIFT features of each frame, each as [N x D]
    """

    def allocate_resource(self, resources, *, num_features=32):
        self.num_features = num_features
        self.sift = None
        return [resources]

    def reset(self):
        if self.sift is None:
            self.sift = cv2.SIFT_create(self.num_features)

    def extract_sift_feature(self, frame: np.ndarray) -> np.ndarray:
        """
        frame: [H x W x C]

        Return: Feature for N key points, [N x 128]
        """
        # TODO: Extract SIFT feature for the current frame
        # Use self.sift.detectAndCompute
        # Remember to handle when it returns None
        raise NotImplementedError

    def process(self, task):
        task.start(self)
        frames = task.content
        frame_ids = task.meta['frame_ids']
        for frame_id, frame in zip(frame_ids, frames):
            sub_task = Task(meta={'sequence_id': frame_id},
                            parent_task=task).start(self)
            feature = self.extract_sift_feature(frame.numpy())
            assert feature is not None and isinstance(feature, np.ndarray)
            assert feature.shape[1] == 128
            yield sub_task.finish(feature)
        task.finish()
