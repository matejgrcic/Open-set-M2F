import torch
from torch import nn
import torch.nn.functional as F
from collections.abc import Iterable
from denseflow.distributions import Distribution
from denseflow.transforms import Transform


class Flow(Distribution):
    """
    Base class for Flow.
    Flows use the forward transforms to transform data to noise.
    The inverse transforms can subsequently be used for sampling.
    These are typically useful as generative models of data.
    """

    def __init__(self, base_dist, transforms, coef=1.):
        super(Flow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.lower_bound = any(transform.lower_bound for transform in transforms)
        self.coef = coef


    def log_prob(self, x, return_z=False):

        log_prob = torch.zeros(x.shape[0], device=x.device)
        for transform in self.transforms:
            x, ldj = transform(x)
            log_prob += ldj
        # log_prob = log_prob / self.base_dist.scale
        log_prob += self.base_dist.log_prob(x)
        log_prob = log_prob / self.coef

        if return_z:
            return x, log_prob
        return log_prob

    def sample(self, num_samples):
        z = self.base_dist.sample(num_samples)
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z

    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")
