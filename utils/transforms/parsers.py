import imgaug
import utils
from utils.transforms import Compose
from typing import List


def dict_list_to_compose(transforms: List[dict]=None):
    r"""Turns list of dictionaries into a Compose.

    Args:
        transforms (list of dict): List of transforms

    Example:
        transforms = [{'Crop': ['top_left_corner': [0, 0], 'shape': [480, 640]]}]
        compose = dict_list_to_compose(transforms)
    """
    if not transforms:
        return None
    compose = []
    for t in transforms:
        (k, kwargs), = t.items()
        compose.append(getattr(utils.transforms, k)(**kwargs))
    return Compose(compose)


def dict_list_to_augment_image(transforms: List[dict]=None):
    r"""Turns list of dictionaries into imgaug augment image, 
    which is a callable.

    Args:
        transforms (list of dict): List of transforms

    Example:
        transforms = [{'chance': 0.5, 'type': , 'kwargs': {}}]
        augment_image = dict_list_to_augment_image(transforms)
        img = augment_image(img)
    """
    if not transforms:
        return None
    augs = imgaug.augmenters.Sequential()
    for t in transforms:
        aug = getattr(eval(t['module']), t['type'])(**t['kwargs'])
        augs.append(imgaug.augmenters.Sometimes(t['chance'], aug))
    return augs.augment_image
