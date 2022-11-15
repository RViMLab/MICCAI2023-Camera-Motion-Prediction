import numpy as np
import importlib
import imgaug
from torchvision.transforms import Compose
from typing import List


def recursive_method_call_from_dict_list(x: object, transforms: List[dict]=None, module: object=None) -> object:
    r"""Performs a recursive call on x.

    Args:
        x (object): Object to forward through the transform
        transforms (list of dict): List of transforms
        module (object): Python module to load transforms from

    Return:
        x (object): Forwarded object
    """
    if not module:
        raise ValueError('Module has to be parsed')
    for t in transforms:
        (k, kwargs), = t.items()
        x = getattr(module, k)(x, **kwargs)
    return x


def dict_list_to_compose(transforms: List[dict]=None, module: object=None) -> Compose:
    r"""Turns list of dictionaries into a Compose.

    Args:
        transforms (list of dict): List of transforms
        module (object): Python module to load transforms from

    Example:
        import utils

        transforms = [{'Crop': ['top_left_corner': [0, 0], 'shape': [480, 640]]}]
        compose = dict_list_to_compose(transforms, utils.transforms)
    """
    if not module:
        raise ValueError('Module has to be parsed')
    if not transforms:
        return None
    compose = []
    for t in transforms:
        (k, kwargs), = t.items()
        compose.append(getattr(module, k)(**kwargs))
    return Compose(compose)

def any_dict_list_to_compose(transforms: List[dict]=None) -> Compose:
    r"""Turns list of dictionaries into a Compose.

    Args:
        transforms (List[dict]): List of transforms, e.g. [{'module': 'some.module', 'type': 'callable', 'kwargs': {'key': val}}]
    Return:
        compose (torchvision.transform.Compose): Callable compose
    """
    if not transforms:
        return None
    compose = []
    for t in transforms:
        try:
            module = importlib.import_module(t['module'])
        except:
            raise ValueError('Parsed module could not be found.')
        compose.append(getattr(module, t['type'])(**t['kwargs']))
    return Compose(compose)


def dict_list_to_augment(transforms: List[dict]=None) -> np.array:
    r"""Turns list of dictionaries into imgaug augment image, 
    which is a callable.

    Args:
        transforms (list of dict): List of transforms

    Example:
        transforms = [{'chance': 0.5, 'type': , 'kwargs': {}, 'module': imgaug.augmenters}]
        augment_image = dict_list_to_augment(transforms)
        img = augment_image(img)
    """
    if not transforms:
        return None
    augs = imgaug.augmenters.Sequential()
    for t in transforms:
        aug = getattr(eval(t['module']), t['type'])(**t['kwargs'])
        augs.append(imgaug.augmenters.Sometimes(t['chance'], aug))
    return augs.augment_image
