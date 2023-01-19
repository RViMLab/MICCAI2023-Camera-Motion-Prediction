from typing import List

import cv2
import numpy as np
from shapely.geometry import MultiPoint, Point


class RandomEdgeHomography:
    r"""Generate random homographies via edge pertubations.

    Implements figure 3 of Deep Homography Estimation, https://arxiv.org/pdf/1606.03798.pdf.

    Args:
        img (np.array): Input image of shape HxWxC
    """

    def __init__(self, img):
        self.img = img

    def compute(self, rho, crp_shape, p0=0.0, verbose=False, max_rollouts: int = 1000):
        r"""Compute the random homographies.

        Args:
            rho (int): uv are perturbed within [-rho, rho]
            crp_shape (tuple of in): Crop shape
            p0 (float): Chance for homography being identity
            verbose (bool): Return dictionary with additional outputs
            max_rollouts (int): Maximum number of homography rollouts

        Return:
            img_crp (np.array): Cropped image of shape crp_shape x C
            wrp_crp (np.array): Cropped warp of shape crp_shape x C
            duv (np.array): Edge displacement of shape 4x2

            if verbose == True:
                H (np.array): Homography of shape 3x3
                uv (np.array): uv coordinates of shape 4x2
                wrp_uv (np.array): Warped uv coordinates of shape 4x2
                img (np.array): Original image
                wrp (np.array): Warped image
                wrp_bdr (np.array): uv of warped image
        """
        feasible = False
        outer_uv = self._shape_to_uv(img.shape[:2])
        outer_shape = img.shape

        rollouts = 0
        p0_sample = np.random.rand()

        while not feasible:
            if rollouts >= max_rollouts or p0_sample < p0:
                # Step 2: Set perturbation to zero
                duv = np.zeros([4, 2], dtype=np.int)

                # Randomly find top left corner that fits crop
                top_left = self._random_top_left(
                    inner_shape=crp_shape, outer_shape=outer_shape[:2]
                )
                inner_uv = self._shape_to_uv(crp_shape, top_left)
                wrp_inner_uv = inner_uv + duv

                # Step 3: Compute homography
                H = cv2.getPerspectiveTransform(
                    inner_uv[:, ::-1].astype(np.float32),
                    wrp_inner_uv[:, ::-1].astype(np.float32),
                )

                # Additional step: Compute outer boarder
                wrp_outer_uv = cv2.perspectiveTransform(
                    outer_uv.reshape(-1, 1, 2)[:, :, ::-1], np.linalg.inv(H)
                )
                feasible = True
                break

            rollouts += 1

            # Step 2: Randomly perturb uv
            duv = np.random.randint(-rho, rho, [4, 2])

            # Randomly find top left corner that fits crop
            top_left = self._random_top_left(
                inner_shape=crp_shape, outer_shape=outer_shape[:2]
            )
            inner_uv = self._shape_to_uv(crp_shape, top_left)
            wrp_inner_uv = inner_uv + duv

            # Step 3: Compute homography
            H = cv2.getPerspectiveTransform(
                inner_uv[:, ::-1].astype(np.float32),
                wrp_inner_uv[:, ::-1].astype(np.float32),
            )

            # Additional step: Check if crop lies within warped image
            wrp_outer_uv = cv2.perspectiveTransform(
                outer_uv.reshape(-1, 1, 2)[:, :, ::-1], np.linalg.inv(H)
            )
            feasible = self._inside(inner_uv, wrp_outer_uv.reshape(-1, 2)[:, ::-1])

        # Step 4: Apply inverse homography to image and crop
        wrp = cv2.warpPerspective(
            img, np.linalg.inv(H), (img.shape[1], img.shape[0])
        ).reshape(outer_shape)
        wrp_crp = self.crop(wrp, inner_uv)
        img_crp = self.crop(img, inner_uv)

        if verbose == True:
            return {
                "img_crp": img_crp,
                "wrp_crp": wrp_crp,
                "duv": duv,
                "H": H,
                "uv": inner_uv,
                "wrp_uv": wrp_inner_uv,
                "img": img.copy(),
                "wrp": wrp,
                "wrp_bdr": wrp_outer_uv,
            }
        else:
            return img_crp, wrp_crp, duv

    def crop(self, img: np.array, uv: np.array):
        r"""Performs crop on image.

        Args:
            img (np.array): Input image of shape HxWxC
            uv (np.array): Edges for crop
        Return:
            crp (np.array): Cropped image
        """
        crp = img[int(uv[0, 0]) : int(uv[2, 0]) + 1, int(uv[0, 1]) : int(uv[2, 1]) + 1]
        return crp

    def visualize(self, dic):
        r"""Creates a visualization of a random homgraphy.

        Args:
            dic (dictionray): Dictionary as returnd by compute()

        Example:
            dic = reh.compute(rho, crp_shape, verbose=True)
            dic = reh.visualize(dic)

            cv2.imshow('wrp', dic['wrp'])
            cv2.imshow('img_crp', dic['img_crp'])
            cv2.imshow('wrp_crp', dic['wrp_crp'])
            cv2.waitKey()
        """
        cv2.polylines(
            dic["img"],
            [dic["uv"][:, ::-1].astype(np.int32)],
            isClosed=True,
            color=(255, 0, 255),
            thickness=2,
        )
        cv2.polylines(
            dic["img"],
            [dic["wrp_uv"][:, ::-1].astype(np.int32)],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
        )
        cv2.polylines(
            dic["wrp"],
            [dic["uv"][:, ::-1].astype(np.int32)],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
        )
        cv2.polylines(
            dic["wrp"],
            [dic["wrp_bdr"].astype(np.int32)],
            isClosed=True,
            color=(255, 255, 0),
            thickness=2,
        )
        return dic

    def _shape_to_uv(self, shape: List[int], top_left: List[int] = [0, 0]):
        r"""Determines the edges of a rectanlge, given the shape and the top left corner.

        Args:
            shape (list of int): Shape of rectangle, HxW
            top_left (list of int): Top left corner of rectangle

        Returns:
            uv (np.array): Edges, 4x2
        """
        uv = np.array(
            [
                [top_left[0], top_left[1]],
                [top_left[0], top_left[1] + shape[1] - 1],
                [top_left[0] + shape[0] - 1, top_left[1] + shape[1] - 1],
                [top_left[0] + shape[0] - 1, top_left[1]],
            ],
            dtype=float,
        )
        return uv

    def _random_top_left(self, inner_shape: List[int], outer_shape: List[int]):
        r"""Determines a random top left corner, which still fits inner rectangle
        into outer rectangle.

        Args:
            inner_shape (list of int): Shape of inner rectangle, HxW
            outer_shape (list of int): Shape of outer rectangle, HxW

        Return:
            top_left (np.array): Random top left corner within feasible area
        """
        top_left = np.random.randint(0, np.subtract(outer_shape, inner_shape), 2)
        return top_left

    def _inside(self, pts, polygon):
        r"""Determine if points lie within a polygon.
        Args:
            pts (array): Points of shape [A, 2 or 3]
            polygon (array): Points of shape [B, 2 or 3]
        """
        inside = True
        n_pts = polygon.shape[0]
        polygon = MultiPoint(polygon).convex_hull
        if (
            len(polygon.exterior.coords) is not n_pts + 1
        ):  # return false in case polygon is not properly computed
            return False
        for p in pts:
            p = Point(p)
            inside = inside and p.within(polygon)
            if not inside:
                break
        return inside


if __name__ == "__main__":
    import os

    path = os.getcwd()
    file_path = os.path.join(path, "utils/processing/sample.npy")
    crp_shape = (128, 128)
    rho = 64

    img = np.load(file_path)
    reh = RandomEdgeHomography(img)

    for i in range(100):
        dic = reh.compute(rho, crp_shape, verbose=True)
        dic = reh.visualize(dic)

        cv2.imshow("img", dic["img"])
        cv2.imshow("wrp", dic["wrp"])
        cv2.imshow("img_crp", dic["img_crp"])
        cv2.imshow("wrp_crp", dic["wrp_crp"])
        cv2.waitKey()
