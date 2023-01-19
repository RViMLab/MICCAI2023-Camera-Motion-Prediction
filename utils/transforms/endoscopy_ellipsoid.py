from typing import List, Tuple

import numpy as np
from skimage.draw import ellipse


class EndoscopyEllipsoid(object):
    def movingCenterPipeline(
        self,
        img: np.array,
        wrp: np.array,
        c_off_scale: List[float] = [0.125, 0.125],
        min_scale: List[float] = [0.3, 0.3],
        max_scale: List[float] = [1.0, 1.0],
        min_rot: float = 0.0,
        max_rot: float = 2 * np.pi,
        dc_scale: List[float] = [0.1, 0.1],
        dori: List[float] = [-np.pi * 0.1, np.pi * 0.1],
        update_chance: float = 1.0,
        seed: np.int32 = None,
    ):
        r"""Builds a pipeline that simulates a moving camera on the endoscope. Adds noise
        around endoscopic view.

        Args:
            img (np.array): Image
            wrp (np.array): Warped image
            c_off_scale (List[float]): Center offset scale of image shape. Perturbes endoscopic view around image center
            min_scale (List[float]): Ellipsoid's half axes minimum scale
            max_scale (List[float]): Ellipsoid's half axes maximum scale
            min_rot (float): Ellipsoid's minimum rotation
            max_rot (float): Ellipsoid's maximum roation
            dc_scale (List[float]): Center update scale, center is perturbed by img.shape*dc_scale
            dori (List[float])
            update_chance (float): Chance by which ellipsoid's center and orientation are updated
            seed (np.int32): Seed for deterministic output

        Return:
            img (np.array): Initial image with random endoscopic ellipsoid and noise surrounding
            wrp (np.array): Initial warp with endoscopic view as for img, but possibly disturbed center
        """
        center = EndoscopyEllipsoid.randomCenter(
            shape=img.shape, scale=c_off_scale, seed=seed
        )
        half_axes = EndoscopyEllipsoid.randomHalfAxes(
            shape=img.shape, min_scale=min_scale, max_scale=max_scale, seed=seed
        )  # assure same half axes
        rot = EndoscopyEllipsoid.randomRot(seed=seed, low=min_rot, high=max_rot)

        img = self(img, center=center, half_axes=half_axes, rot=rot)

        # update center by chance
        center, rot = EndoscopyEllipsoid.randomMotion(
            img.shape,
            center=center,
            rot=rot,
            scale=dc_scale,
            dori=dori,
            chance=update_chance,
            seed=seed,
        )
        wrp = self(wrp, center=center, half_axes=half_axes, rot=rot)

        return img, wrp

    @staticmethod
    def randomMotion(
        shape: Tuple[int],
        center: Tuple[int],
        rot: float,
        scale: List[float] = [0.1, 0.1],
        dori: List[float] = [-np.pi * 0.1, np.pi * 0.1],
        chance: float = 1.0,
        seed: np.int32 = None,
    ) -> Tuple[np.array, float]:
        r"""Helper function to randomly update center and rotation.

        Args:
            shape (Tuple[int]): Image shape HxWxC
            center (Tuple[int]): Initial endoscopic view center
            rot (float): Initial rotation
            scale (List[float]): Center update scale
            dori (List[float]): Orientation delta sample interval
            chance (float): Chance by which center is updated
            seed (np.int32): Seed for deterministic output

        Return:
            new_center (Tuple[np.array, float]): Updated center
        """
        np.random.seed(seed)
        if np.random.rand(1) <= chance:
            new_center = np.random.uniform(
                center - np.array(shape)[:2] * scale,
                center + np.array(shape)[:2] * scale,
                2,
            )

            new_rot = rot + np.random.uniform(dori[0], dori[1], 1)[0]
            np.random.seed(None)
            return new_center, new_rot
        else:
            np.random.seed(None)
            return center, rot

    @staticmethod
    def randomCenter(
        shape: Tuple[int], scale: List[float] = [0.125, 0.125], seed: np.int32 = None
    ) -> np.array:
        r"""Helper function to generate random center.

        Args:
            shape (Tuple[int]): Image shape HxWxC
            scale (List[float]): Center offset scale of image shape. Perturbes endoscopic view around image center
            seed (np.int32): Seed for deterministic output

        Return:
            center (np.array): Center of endoscopic view
        """
        np.random.seed(seed)
        shape = np.array(shape)
        center = np.random.uniform(
            shape[:2] / 2 - 1 - shape[:2] * scale, shape[:2] / 2 - 1 + shape[:2] * scale
        )
        np.random.seed(None)

        return center

    @staticmethod
    def randomHalfAxes(
        shape: Tuple[int],
        min_scale: List[float] = [0.3, 0.3],
        max_scale: List[float] = [1.0, 1.0],
        seed: np.int32 = None,
    ) -> np.array:
        r"""Compute random half axes. Returns uniformly sampled half axes in [min_scale, max_scale]*shape[:2].

        Args:
            shape (Tuple[int]): Image shape HxWxC
            min_scale (List[float]): Minimum scale
            max_scale (List[float]): Maximum scale
            seed (np.int32): Seed for deterministic output
        Return:
            half_axes (np.array): Ellipsoid's half axes
        """
        np.random.seed(seed)
        half_axes = np.random.uniform(min_scale, max_scale, 2) * shape[:2]
        np.random.seed(None)

        return half_axes

    @staticmethod
    def randomRot(
        seed: np.int32 = None, low: float = 0, high: float = 2 * np.pi
    ) -> float:
        r"""Compute random rotation.

        Args:
            seed (np.int32): Seed for deterministic output
            low (float): Lower uniform boundary
            high (float): Upper uniform boundary
        Return:
            rot (float): Random rotation in [low, high]
        """
        np.random.seed(seed)
        rot = np.random.uniform(low, high, 1)
        np.random.seed(None)
        return rot[0]

    def __call__(
        self,
        img: np.array,
        center: np.array = None,
        half_axes: np.array = None,
        rot: float = None,
    ):
        r"""Puts a elliptic noisy mask on top of an image.

        Args:
            img (np.array): uint8 image to be masked of shape HxWxC
            center (np.array): Ellipsoid's center
            half_axes (np.array): Ellipsoid's half axes
            rot (float): Ellipsoid's rotation
        """
        return self._ellipticNoisyMask(img, center=center, half_axes=half_axes, rot=rot)

    def _ellipticNoisyMask(
        self,
        img: np.array,
        center: np.array = None,
        half_axes: np.array = None,
        rot: float = None,
    ) -> np.array:
        r"""Puts a elliptic noisy mask on top of an image.

        Args:
            img (np.array): uint8 image to be masked of shape HxWxC
            center (np.array): Circular mask's center
            half_axes (np.array): Circular mask's radius

        Return:
            img (np.array): Masked image
        """
        rr, cc = ellipse(
            r=center[0],
            c=center[1],
            r_radius=half_axes[0],
            c_radius=half_axes[1],
            rotation=rot,
            shape=img.shape,
        )

        noise = np.random.uniform(0, 255, img.shape).astype(np.uint8)
        mask = np.zeros_like(img, dtype=bool)
        mask[rr, cc] = True
        img = np.where(mask, img, noise)

        return img


if __name__ == "__main__":
    import cv2

    pipe = True

    ec = EndoscopyEllipsoid()

    img = np.load("utils/transforms/sample.npy")

    # sample 300 loops
    if pipe:
        for _ in range(300):
            img0, img1 = ec.movingCenterPipeline(img, img)
            cv2.imshow("img0", img0)
            cv2.imshow("img1", img1)
            cv2.waitKey()
    else:
        for _ in range(300):
            center = EndoscopyEllipsoid.randomCenter(img.shape)
            half_axes = EndoscopyEllipsoid.randomHalfAxes(img.shape, seed=5)
            rot = EndoscopyEllipsoid.randomRot(seed=5)

            img0 = ec(img, center=center, half_axes=half_axes, rot=rot)

            half_axes = ec.randomHalfAxes(img.shape)
            rot = EndoscopyEllipsoid.randomRot()

            img1 = ec(img, center=center, half_axes=half_axes, rot=rot)

            # random update
            center, rot = EndoscopyEllipsoid.randomMotion(
                img.shape, center=center, rot=rot
            )
            img2 = ec(img, center=center, half_axes=half_axes, rot=rot)

            cv2.imshow("img0", img0)
            cv2.imshow("img1", img1)
            cv2.imshow("img2", img2)
            cv2.waitKey()
