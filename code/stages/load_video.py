import os.path as osp
import numpy as np
import torch

from pyturbo import Stage, Task
from torchvision.io import read_video


class LoadVideo(Stage):

    """
    Input: video_id
    Output: yield batches of selected frames, each batch as 
        [B x H x W x C]
    """

    def allocate_resource(self, resources, *, video_dir, file_suffix='mp4',
                          worker_per_cpu=1, batch_size=1, 
                          target_frame_rate: float = 1.5):
        self.video_dir = video_dir
        self.file_suffix = file_suffix
        self.batch_size = batch_size
        self.target_frame_rate = target_frame_rate
        return resources.split(len(resources.get('cpu'))) * worker_per_cpu

    def downsample_frames(self, frames: np.ndarray, frame_rate: float):
        """
        frames: [T x H x W x C]
        frame_rate: number of frames per second

        Return: downsampled frames [t x H x W x C]
        """
        # TODO: downsample the frames to self.target_frame_rate
        raise NotImplementedError

    def process(self, task):
        task.start(self)
        video_id = task.content
        video_path = osp.join(self.video_dir, f'{video_id}.{self.file_suffix}')
        frames, _, meta = read_video(video_path, pts_unit='sec')
        frames = frames.numpy()
        frame_rate = meta['video_fps']
        selected_frames = self.downsample_frames(frames, frame_rate)
        for i, start in enumerate(range(
                0, selected_frames.shape[0], self.batch_size)):
            batch_frames = selected_frames[start:start + self.batch_size]
            frame_ids = [*range(start, start + batch_frames.shape[0])]
            sub_task = Task(meta={'frame_ids': frame_ids, 'batch_id': i},
                            parent_task=task).start(self)
            yield sub_task.finish(torch.as_tensor(batch_frames))
        task.finish()
