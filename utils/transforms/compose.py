import utils
from typing import List


class Compose(object):
    def __init__(self, transforms: List[object]):
        r"""Composes a list o transforms.

        Args: 
            transforms (list of callables): Transforms to be composed
        """
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def dict_list_to_compose(transforms: List[dict]):
    r"""Turns list of dictionaries into a Compose.

    Args:
        transforms (list of dict): List of transforms

    Example:
        transforms = [{'Crop': ['top_left_corner': [0, 0], 'shape': [480, 640]]}]
        compose = dict_list_to_compose(transforms)
    """
    compose = []
    for t in transforms:
        (k, kwargs), = t.items()
        compose.append(getattr(utils.transforms, k)(**kwargs))
    return Compose(compose)
