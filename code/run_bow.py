import argparse
import os.path as osp

import pandas as pd
from pyturbo import Job, Options, System, Task

from stages import BagOfWords, LoadFeature, SaveFeature


class BuildBagOfWords(System):

    def get_num_pipeline(self, resources, *, args):
        self.args = args
        return min(8, len(resources.get('cpu')))

    def get_stages(self, resources):
        stages = [
            LoadFeature(resources, feature_dir=self.args.feature_dir),
            BagOfWords(resources, weight_path=self.args.weight_path),
            SaveFeature(resources, feature_dir=self.args.bow_dir),
        ]
        return stages


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument('list_file_path')
    parser.add_argument('model_name')
    parser.add_argument('feature_dir')
    parser.add_argument(
        '--model_dir', default=osp.join(
            osp.dirname(__file__), '../data/kmeans'))
    parser.add_argument(
        '--bow_dir_prefix', default=osp.join(osp.dirname(__file__), 
        '../data/bow'))
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
    args.weight_path = osp.join(args.model_dir, f'{args.model_name}.pkl')
    args.bow_dir = f'{args.bow_dir_prefix}_{args.model_name}'
    system = BuildBagOfWords(args=args)
    system.start()
    jobs = build_jobs(args)
    system.add_jobs(jobs)
    try:
        for job in system.wait_jobs(len(jobs)):
            continue
        system.end()
    except:
        system.terminate()


if __name__ == '__main__':
    main(parse_args())
