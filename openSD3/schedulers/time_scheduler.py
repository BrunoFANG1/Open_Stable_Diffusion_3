import torch
from abc import ABC, abstractmethod
import numpy as np
import torch as th

### This file is mostly adopted from OpenSORA project
### I add a new Logit-Normal Samping in this paper Scaling Rectified Flow Transformers for High-Resolution Image Synthesis


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.
    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "logitnorm":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.
    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.
        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.
        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights

class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

class LogitNormalSampler(ScheduleSampler):
    def __init__(self, diffusion, m, s):
        self.diffusion = diffusion
        self.m = m 
        self.s = s
    
    def weights(self):
        t = (np.arange(self.diffusion.num_timesteps)+0.01)/self.diffusion.num_timesteps #[0-1]
        w = 1/(self.s*np.sqrt(2*np.pi)) * 1/(t-t*t)*np.exp(-(np.log(t/(1-t))-self.m)**2/(2*self.s**2))
        return w


def test():
    
    class diffusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_timesteps = 100

    DF = diffusion()
    sampler = LogitNormalSampler(DF, 0, 0.5)
    indices, weights = sampler.sample(5, torch.device('cpu'))
    print(indices)
    print(weights)

if __name__ == "__main__":
    test()

