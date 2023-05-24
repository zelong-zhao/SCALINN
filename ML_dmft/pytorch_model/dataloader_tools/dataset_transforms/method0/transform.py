import torch
import numpy as np
from torchvision import transforms


class ToTensor:
    def __init__(self, precision=None):
        self.precision = precision
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        if self.precision:
            return torch.from_numpy(inputs).type(self.precision), torch.from_numpy(targets).type(self.precision)
        else:
            return torch.from_numpy(inputs), torch.from_numpy(targets)

class Squasher:
    # multiply inputs with a given factor
    def __init__(self, factor=None):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        if self.factor is not None:
            inputs += self.factor
            inputs = np.log(inputs)
        return inputs, targets

class Normalise:
    def __init__(self, norm_factor=1):
        self.norm_factor = max([norm_factor,1e-12])

    def __call__(self,sample):
        inputs, targets = sample
        inputs /= self.norm_factor
        return inputs, targets

class Squasher_Y:
    # multiply inputs with a given factor
    def __init__(self, factor=None):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        if self.factor is not None:
            targets += self.factor
            targets = np.log(targets)
        return inputs, targets


class Normalise_Y:
    def __init__(self, norm_factor=1):
        self.norm_factor = max([norm_factor,1e-12])

    def __call__(self,sample):
        inputs, targets = sample
        targets /= self.norm_factor
        return inputs, targets


def stacked_transform_xy(squasher_factor,normalise_factor,squasher_factor_y,normalise_factor_y):
    transform=transforms.Compose([Squasher(squasher_factor),
                                Squasher_Y(squasher_factor_y),
                                Normalise(normalise_factor),
                                Normalise_Y(normalise_factor_y),
                                ToTensor(torch.float)])
    transform=transforms.Compose([ToTensor(torch.float)])
    return transform

class Y_Inverse_Transform():
    def __init__(self,squasher_factor_y=None,normalise_factor_y=1) -> None:
        self.squasher_factor_y=squasher_factor_y
        self.normalise_factor_y=normalise_factor_y

    def __call__(self,tranformed_target):
        target=np.array(tranformed_target)
        target=target*self.normalise_factor_y
        if self.squasher_factor_y is not None:
            target=np.exp(target)
            target-=self.squasher_factor_y
        return target