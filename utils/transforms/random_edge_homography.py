import cv2
import numpy as np
from enum import IntEnum
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from typing import List, Tuple


class HOMOGRAPHY_RETURN(IntEnum):
    DEFAULT = 1
    VISUAL = 2
    DATASET = 3


class RandomEdgeHomography(object):
    r"""Callable random homographies generator via edge pertubations.

    Implements figure 3 of Deep Homography Estimation, https://arxiv.org/pdf/1606.03798.pdf.

    Args:
        rho (int): uv are perturbed within [-rho, rho]
        crp_shape (tuple of int): Crop shape
        homography_return (IntEnum): Return different outputs on __call__()
        seeds (list of np.int32): Seeds for deterministic output, list of randomly generated int
    """
    def __init__(self, rho: int, crp_shape: List[int], homography_return: IntEnum=HOMOGRAPHY_RETURN.DEFAULT, seeds: List[np.int32]=None):
        self.rho = rho
        self.crp_shape = crp_shape
        self.homography_return = homography_return
        self.seeds = seeds

    def __call__(self, img: np.array, idx: int):
        """Compute the random homographies.

        Args:
            img (np.array): Input image of shape HxWxC
            idx (int): Index for self.seeds

        Return:
            img_crp (np.array): Cropped image of shape crp_shape x C
            wrp_crp (np.array): Cropped warp of shape crp_shape x C
            duv (np.array): Edge displacement of shape 4x2

            if return_dic == True:
                H (np.array): Homography of shape 3x3
                uv (np.array): uv coordinates of shape 4x2
                wrp_uv (np.array): Warped uv coordinates of shape 4x2
                img (np.array): Original image
                wrp (np.array): Warped image
                wrp_bdr (np.array): uv of warped image
        """
        # retrieve seed from list of seeds
        if self.seeds:
            seed = self.seeds[idx]
            np.random.seed(seed)
        img_crp, uv = self._random_crop(img=img, crp_shape=self.crp_shape, padding=self.rho)

        feasible = False
        boundary = np.array([
                [0               ,                0],
                [0               , img.shape[1] - 1],
                [img.shape[0] - 1, img.shape[1] - 1],
                [img.shape[0] - 1,                0],
        ], dtype=np.float)
        shape = img.shape

        while not feasible:
            # Step 2: Randomly perturb uv
            duv = np.random.randint(-self.rho, self.rho, [4,2])
            wrp_uv = uv + duv

            # Step 3: Compute homography
            H = cv2.getPerspectiveTransform(uv[:,::-1].astype(np.float32), wrp_uv[:,::-1].astype(np.float32))

            # Additional step: Check if crop lies within warped image
            wrp_bdr = cv2.perspectiveTransform(boundary.reshape(-1,1,2)[:,:,::-1], np.linalg.inv(H))
            feasible = self._inside(uv, wrp_bdr.reshape(-1,2)[:,::-1])

        # Step 4: Apply inverse homography to image and crop
        wrp = cv2.warpPerspective(img, np.linalg.inv(H), (img.shape[1], img.shape[0])).reshape(shape)
        wrp_crp = self.crop(wrp, uv)

        np.random.seed(None)
            
        if self.homography_return == HOMOGRAPHY_RETURN.DEFAULT:
            return img_crp, wrp_crp, duv
        if self.homography_return == HOMOGRAPHY_RETURN.VISUAL:
            return {
                'img_crp': img_crp, 
                'wrp_crp': wrp_crp, 
                'duv': duv,
                'H': H,
                'uv': uv, 
                'wrp_uv': wrp_uv,
                'img': img.copy(),
                'wrp': wrp, 
                'wrp_bdr': wrp_bdr
            }
        if self.homography_return == HOMOGRAPHY_RETURN.DATASET:
            return {
                'img': img.copy(),
                'img_crp': img_crp, 
                'wrp_crp': wrp_crp, 
                'uv': uv,
                'duv': duv,
                'H': H
            }
            

    def visualize(self, dic: dict):
        r"""Creates a visualization of a random homgraphy.

        Args:
            dic (dictionray): Dictionary as returnd by self.__call__()

        Example:
            reh = RandomEdgeHomography(rho, crp_shape, homography_return=True)
            dic = reh(img)
            dic = reh.visualize(dic)

            cv2.imshow('wrp', dic['wrp'])
            cv2.imshow('img_crp', dic['img_crp'])
            cv2.imshow('wrp_crp', dic['wrp_crp'])
            cv2.waitKey()
        """
        cv2.polylines(dic['img'], [dic['uv'][:,::-1].astype(np.int32)], isClosed=True, color=(255, 0, 255), thickness=2)
        cv2.polylines(dic['img'], [dic['wrp_uv'][:,::-1].astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.polylines(dic['wrp'], [dic['uv'][:,::-1].astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.polylines(dic['wrp'], [dic['wrp_bdr'].astype(np.int32)], isClosed=True, color=(255, 255, 0), thickness=2)
        return dic

    def _inside(self, pts: np.array, polygon: np.array):
        r"""Determine if points lie within a polygon.
        Args:
            pts (np.array): Points of shape [A, 2 or 3]
            polygon (np.array): Points of shape [B, 2 or 3]
        """
        inside = True
        n_pts = polygon.shape[0]
        polygon = MultiPoint(polygon).convex_hull
        if len(polygon.exterior.coords) is not n_pts + 1:  # return false in case polygon is not properly computed
            return False
        for p in pts:
            p = Point(p)
            inside = inside and p.within(polygon)
            if not inside:
                break
        return inside

    def _random_crop(self, img: np.array, crp_shape: Tuple[int], padding: Tuple[int]):
        r"""Performs random image crop.

        Args:
            img (np.array): Input image of shape HxWxC
            crp_shape (tuple of int): Destination array shape
            padding (int or tuple of int): Padding area that is not cropped from
        Return:
            crp (np.array): Cropped image
            uv (np.array): Edge coordinates
        """
        top_left = np.random.randint(padding, np.subtract(img.shape[:2], np.add(padding, crp_shape)), 2)
        uv = np.array([
            top_left,
            [top_left[0]               , top_left[1] + crp_shape[1]],
            [top_left[0] + crp_shape[0], top_left[1] + crp_shape[1]],
            [top_left[0] + crp_shape[0], top_left[1]               ]
        ])
        crp = self.crop(img, uv)
        return crp, uv

    def crop(self, img: np.array, uv: np.array):
        r"""Performs crop on image.

        Args:
            img (np.array): Input image of shape HxWxC
            uv (np.array): Edges for crop
        Return:
            crp (np.array): Cropped image
        """
        crp = img[uv[0,0]:uv[2,0],uv[0,1]:uv[2,1]]
        return crp


if __name__ == '__main__':
    import os
    path = os.getcwd()
    file_path = os.path.join(path, 'utils/transforms/sample.npy')
    crp_shape = (128, 256)
    rho = 64

    img = np.load(file_path)
    reh = RandomEdgeHomography(rho, crp_shape, True)

    for i in range(100):
        dic = reh(img)
        dic = reh.visualize(dic)

        cv2.imshow('img', dic['img'])
        cv2.imshow('wrp', dic['wrp'])
        cv2.imshow('img_crp', dic['img_crp'])
        cv2.imshow('wrp_crp', dic['wrp_crp'])
        cv2.waitKey()
