import argparse
import os.path as osp

import pandas as pd
from pyturbo import Job, Options, System, Task

from stages import CNN3DFeature, LoadVideo, SaveFeature


class ExtractCNN3DFeature(System):

    def get_num_pipeline(self, resources, *, args):
        """Returns desired number of pipelines to run in parallel."""
        self.args = args
        self.resources = resources.select(cpu=True)
        # Manual allocation of GPU resources for multi-pipeline per GPU setup
        gpu_resources = resources.select(gpu=True).split(
            len(resources.get('gpu')))
        self.gpu_resources = gpu_resources * self.args.pipeline_per_gpu
        if len(self.gpu_resources) == 0:
            # CPU only
            return max(1, len(resources.get('cpu')) // 2)
        else:
            return len(self.gpu_resources)

    def get_stages(self, resources):
        """Returns stages for the current pipeline. 
           (Called once for each pipeline)"""
        io_resources = resources.select(cpu=(0, 1))
        cnn_resources = resources + self.gpu_resources.pop(0)
        stages = [
            LoadVideo(io_resources, video_dir=self.args.video_dir,
                      target_frame_rate=self.args.frame_rate,
                      batch_size=self.args.clip_duration),
            CNN3DFeature(cnn_resources,
                         # TODO: choose the model, weight, and node to use
                         model_name='',
                         weight_name='',
                         node_name='',
                         replica_per_gpu=self.args.replica_per_gpu),
            SaveFeature(io_resources, feature_dir=self.args.cnn_dir),
        ]
        return stages


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument('list_file_path')
    parser.add_argument(
        '--video_dir', default=osp.join(
            osp.dirname(__file__), '../data/videos'))
    parser.add_argument(
        '--cnn_dir', default=osp.join(osp.dirname(__file__), '../data/cnn3d'))
    parser.add_argument('--frame_rate', type=int, default=15)
    parser.add_argument('--clip_duration', type=int, default=32)
    parser.add_argument('--pipeline_per_gpu', type=int, default=2)
    parser.add_argument('--replica_per_gpu', type=int, default=1)
    parser.add_argument('--job_timeout', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)
    return args


def build_jobs(args):
    df = pd.read_csv(args.list_file_path)
    video_ids = df['Id']
    jobs = [Job(vid, Task(vid, {'video_id': vid})) for vid in video_ids]
    return jobs


def main(args):
    if args.debug:
        Options.single_sync_pipeline = True
        Options.raise_exception = True
    system = ExtractCNN3DFeature(args=args)
    system.start()
    jobs = build_jobs(args)
    system.add_jobs(jobs)
    try:
        for job in system.wait_jobs(len(jobs), job_timeout=args.job_timeout):
            continue
        system.end()
    except:
        system.terminate()


if __name__ == '__main__':
    main(parse_args())
