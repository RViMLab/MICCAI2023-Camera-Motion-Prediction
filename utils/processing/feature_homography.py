from typing import List, Tuple, Union, TypeVar

import cv2
import kornia
import numpy as np
import torch

from .helpers import image_edges


class FeatureHomographyEstimation(object):
    r"""Homography estimation based on feature detectors.

    Args:
        fd (cv2.Feature2D): Feature detector https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html

    Example:
        fd = cv2.SIFT_create()
        fh = FeatureHomographyEstimation(fd)

        H, duv = fh(img, wrp)

        wrp_est = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    """
    def __init__(self, fd: cv2.Feature2D) -> None:
        self._fd = fd
        self._matcher = cv2.FlannBasedMatcher()

    def __call__(self, img: np.array, wrp: np.array, ransacReprojThreshold=5.0, return_kp: bool=False) -> Union[Tuple[np.array, np.array], Tuple[np.array, np.array, np.array, np.array, np.array]]:
        r"""Estimates the homography between images and returns point representation and homography.

        Args:
            img (np.array): Input image of shape HxWxC
            wrp (np.array): Warped input image of shape HxWxC
            ransacReprojThreshold (double): Ransac reprojection error threshold
            return_kp (bool): If true, return keypoints

        Returns:
            H (np.array): Estimated homography of shape 3x3
            duv (np.array): Deformation of image edges uv under estimated homography 4x2

            if return_kp:
                kp_img (np.array): Matching image keypoints of shape Nx1x2
                kp_wrp (np.array): Matching warp keypoints of shape Nx1x2
        """
        uv = self._image_edges(img)

        img, wrp = self._grayscale(img, wrp)
        kp_img, des_img = self._fd.detectAndCompute(img, None)
        kp_wrp, des_wrp = self._fd.detectAndCompute(wrp, None)
        if des_img is None or des_wrp is None or des_img.shape[0] < 2 or des_wrp.shape[0] < 2:
            if return_kp:
                return None, None, None, None, None
            else:   
                return None, None

        des_img, des_wrp = des_img.astype(np.float32), des_wrp.astype(np.float32)
        good_matches = self._match(des_img, des_wrp)

        kp_img = np.float32([ kp_img[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        kp_wrp = np.float32([ kp_wrp[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        if kp_img.shape[0] < 4 or kp_wrp.shape[0] < 4:
            if return_kp:
                return None, None, None, None, None
            else:   
                return None, None
        H, mask = cv2.findHomography(kp_img, kp_wrp, cv2.RANSAC, ransacReprojThreshold)
        if H is None:
            if return_kp:
                return None, None, None, None, None
            else:   
                return None, None

        uv_pred = cv2.perspectiveTransform(uv.astype(np.float32).reshape(-1, 1, 2)[...,::-1], H).squeeze()
        duv = uv - uv_pred[...,::-1]

        if return_kp:
            return H, duv, kp_img, kp_wrp, mask
        else:
            return H, duv

    def _grayscale(self, img: np.array, wrp: np.array) -> Tuple[np.array, np.array]:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        wrp = cv2.cvtColor(wrp, cv2.COLOR_RGB2GRAY)
        return img, wrp

    def _match(self, queryDescriptors, trainDescriptors) -> List[cv2.DMatch]:
        matches = self._matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

        good_matches = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good_matches.append(m)

        return good_matches

    def _image_edges(self, img: np.array):
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

TLoFTRHomographyEstimation = TypeVar("TLoFTRHomographyEstimation", bound="LoFTRHomographyEstimation")

class LoFTRHomographyEstimation(object):
    def __init__(self) -> None:
        self._loftr = kornia.feature.LoFTR()
        self._ransac = kornia.geometry.RANSAC(inl_th=1.0)    

    def __call__(self, img: torch.Tensor, wrp: torch.Tensor) -> torch.Tensor:
        input = {"image0": kornia.color.rgb_to_grayscale(img), "image1": kornia.color.rgb_to_grayscale(wrp)}
        correspondence_dict = self._loftr(input)

        H = []
        for i in range(img.shape[0]): # ransac might not be parallelizeable!
            try:
                H.append(self._ransac(correspondence_dict["keypoints0"][correspondence_dict["batch_indexes"] == i], correspondence_dict["keypoints1"][correspondence_dict["batch_indexes"] == i])[0])
            except:
                H.append(torch.full([3, 3], torch.nan, device=img.device))
        H = torch.stack(H)
        uv0 = image_edges(img).flip(-1)
        uv1 = kornia.geometry.transform_points(H, uv0)
        return (uv0 - uv1).flip(-1)  # OpenCV convention, this really needs a fix, includes training of deep homography estimation!

    def to(self, device: torch.device) -> TLoFTRHomographyEstimation:
        self._loftr.to(device=device)
        self._ransac.to(device=device)
        return self

    def eval(self) -> TLoFTRHomographyEstimation:
        self._loftr.eval()
        self._ransac.eval()
        return self


if __name__ == "__main__":
    def test_cv_homography():
        fd = cv2.SIFT_create()
        fh = FeatureHomographyEstimation(fd)

        img = np.load("utils/processing/sample.npy")

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

        cv2.imshow("composite", np.concatenate([img, wrp ,wrp_est], axis=1))
        cv2.waitKey()

        print("H - H_pred:\n", H - H_pred)
        print("duv:\n", duv)
    
    def test_kornia_homography():
        import cv2
        import matplotlib.pyplot as plt
        from kornia import image_to_tensor, tensor_to_image
        from kornia.geometry import warp_perspective
        from helpers import four_point_homography_to_matrix, image_edges

        estimator = LoFTRHomographyEstimation()
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        estimator.to(device)

        img = np.load("utils/processing/sample.npy")[:,:,::-1]
        img = cv2.resize(img, [320, 240])

        # create fake warp
        H = np.eye(3)
        H[0, 0] += 1
        wrps = []
        wrps.append(cv2.warpPerspective(img, H, (img.shape[1], img.shape[0])))
        H[0, 1] += 1
        wrps.append(cv2.warpPerspective(img, H, (img.shape[1], img.shape[0])))

        plt.imshow(wrps[0])
        plt.show()
        plt.imshow(wrps[1])
        plt.show()

        # convert to torch
        img = image_to_tensor(img, False)
        for i in range(len(wrps)):
            wrps[i] = image_to_tensor(wrps[i], True)
        imgs = torch.concat([img, img])
        wrps = torch.stack(wrps)
        imgs, wrps = imgs.to(device).float()/255., wrps.to(device).float()/255.

        # estimate warp
        duvs = estimator(imgs, wrps)

        hs = four_point_homography_to_matrix(image_edges(imgs), -duvs)
        wrps_pred = warp_perspective(imgs, hs, imgs.shape[-2:])
        for wrp_pred in wrps_pred:
            wrp_pred = tensor_to_image(wrp_pred, False)
            plt.imshow(wrp_pred)
            plt.show()
        
    # test_cv_homography()
    # test_kornia_homography()

