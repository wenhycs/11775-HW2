import os
import os.path as osp
import pickle

from pyturbo import ReorderStage


class SaveFeature(ReorderStage):

    """
    Input: feature
    Save to file.
    """

    def allocate_resource(self, resources, *, feature_dir, file_suffix='pkl'):
        self.feature_dir = feature_dir
        self.file_suffix = file_suffix
        os.makedirs(self.feature_dir, exist_ok=True)
        return [resources]

    def get_sequence_id(self, task):
        return task.meta.get('sequence_id', 0)

    def process(self, task):
        task.start(self)
        feature = task.content
        video_id = task.meta['video_id']
        sequence_id = task.meta.get('sequence_id', 0)
        feature_path = osp.join(
            self.feature_dir, f'{video_id}.{self.file_suffix}')
        with open(feature_path, 'ab' if sequence_id > 0 else 'wb') as f:
            pickle.dump((sequence_id, feature), f)
        return task.finish()
