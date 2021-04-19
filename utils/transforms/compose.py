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
