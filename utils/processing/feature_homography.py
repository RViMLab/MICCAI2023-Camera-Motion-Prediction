
import cv2
import numpy as np
from typing import Tuple, List


class FeatureHomographyEstimation(object):
    r"""Homography estimation based on feature detectors.

    Args:
        fd (cv2.Feature2D): Feature detector https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html

    Example:
        fd = cv2.xfeatures2d.SIFT_create()
        fh = FeatureHomographyEstimation(fd)

        H, duv = fh(img, wrp)

        wrp_est = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    """
    def __init__(self, fd: cv2.Feature2D) -> None:
        self.fd = fd
        self.matcher = cv2.FlannBasedMatcher()

    def __call__(self, img: np.array, wrp: np.array, ransacReprojThreshold=5.0) -> Tuple[np.array, np.array]:
        r"""Estimates the homography between images and returns point representation and homography.

        Args:
            img (np.array): Input image of shape HxWxC
            wrp (np.array): Warped input image of shape HxWxC
            ransacReprojThreshold (double): Ransac reprojection error threshold

        Returns:
            H (np.array): Estimated homography of shape 3x3
            duv (np.array): Deformation of image edges uv under estimated homography 4x2
        """
        uv = self._imageEdges(img)

        img, wrp = self._grayscale(img, wrp)
        kp_img, des_img = self.fd.detectAndCompute(img, None)
        kp_wrp, des_wrp = self.fd.detectAndCompute(wrp, None)
        if des_img is None or des_wrp is None or des_img.shape[0] < 2 or des_wrp.shape[0] < 2:
            return None, None

        good_matches = self._match(des_img, des_wrp)

        kp_img = np.float32([ kp_img[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        kp_wrp = np.float32([ kp_wrp[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        if kp_img.shape[0] < 4 or kp_wrp.shape[0] < 4:
            return None, None
        H, _ = cv2.findHomography(kp_img, kp_wrp, cv2.RANSAC, ransacReprojThreshold)
        if H is None:
            return None, None

        uv_pred = cv2.perspectiveTransform(uv.astype(np.float32).reshape(-1, 1, 2), H).squeeze()
        duv = uv_pred - uv

        return H, duv

    def _grayscale(self, img: np.array, wrp: np.array) -> Tuple[np.array, np.array]:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        wrp = cv2.cvtColor(wrp, cv2.COLOR_RGB2GRAY)
        return img, wrp

    def _match(self, queryDescriptors, trainDescriptors) -> List[cv2.DMatch]:
        matches = self.matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

        good_matches = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good_matches.append(m)

        return good_matches

    def _imageEdges(self, img: np.array):
        r"""Returns edges of image (uv) in OpenCV convention.

        Args:
            img (np.array): Image of shape HxWxC

        Returns:
            uv (np.array): Image edges of shape 4x2
        """
        shape = img.shape[:2]
        uv = np.array(
            [
                [       0,        0],
                [       0, shape[1]],
                [shape[0], shape[1]],
                [shape[0],        0]
            ]
        )
        return uv


if __name__ == '__main__':
    fd = cv2.xfeatures2d.SIFT_create()
    fh = FeatureHomographyEstimation(fd)

    img = np.load('utils/processing/sample.npy')

    # create fake warp
    H = np.eye(3)
    H[0, 0] += 1
    wrp = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))

    # estimate warp
    H_pred, duv = fh(img, wrp)
    wrp_est = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))

    # plot results
    shape = (int(img.shape[1]/2.), int(img.shape[0]/2.))

    img = cv2.resize(img, shape)
    wrp = cv2.resize(wrp, shape)
    wrp_est = cv2.resize(wrp_est, shape)

    cv2.imshow('composite', np.concatenate([img, wrp ,wrp_est], axis=1))
    cv2.waitKey()

    print('H - H_pred:\n', H - H_pred)
    print('duv:\n', duv)
