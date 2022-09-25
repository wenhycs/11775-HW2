import numpy as np
import torch
from pyturbo import Stage, Task
from torch.backends import cudnn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class CNNFeature(Stage):

    """
    Input: batch of frames [B x H x W x C]
    Output: yield CNN features of each frame, each as [D]
    """

    def allocate_resource(self, resources, *, model_name='resnet18',
                          weight_name='ResNet18_Weights',
                          node_name='avgpool', replica_per_gpu=1):
        self.model_name = model_name
        self.weight_name = weight_name
        self.node_name = node_name
        self.model = None
        gpus = resources.get('gpu')
        self.num_gpus = len(gpus)
        if len(gpus) > 0:
            return resources.split(len(gpus)) * replica_per_gpu
        return [resources]

    def reset(self):
        if self.model is None:
            gpu_ids = self.current_resource.get('gpu', 1)
            if len(gpu_ids) >= 1:
                self.device = 'cuda:%d' % (gpu_ids[0])
                cudnn.fastest = True
                cudnn.benchmark = True
            else:
                self.device = 'cpu'
                self.logger.warn('No available GPUs, running on CPU.')
            weights = getattr(models, self.weight_name).DEFAULT
            self.transforms = weights.transforms()
            base_model = getattr(models, self.model_name)(weights=weights)
            self.model = create_feature_extractor(
                base_model, {self.node_name: 'feature'})
            self.model = self.model.to(self.device).eval()

    def extract_cnn_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frame: [B x H x W x C] in uint8 [0, 255]

        Return: Feature, [B x D]
        """
        # TODO: extract CNN feature for given batch
        # First convert batch into [B x C x H x W] format expected by PyTorch.
        # Then apply self.transforms to batch to get model input.
        # Finally apply self.model on the input to get features.
        # Wrap the model with torch.no_grad() to avoid OOM.
        raise NotImplementedError

    def process(self, task):
        task.start(self)
        frames = task.content
        frame_ids = task.meta['frame_ids']
        features = self.extract_cnn_features(frames).cpu().numpy()
        for frame_id, feature in zip(frame_ids, features):
            sub_task = Task(meta={'sequence_id': frame_id},
                            parent_task=task).start(self)
            yield sub_task.finish(feature)
        task.finish()
