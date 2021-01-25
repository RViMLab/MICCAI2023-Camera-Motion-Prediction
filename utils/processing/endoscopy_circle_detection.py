import cv2
import numpy as np
from typing import Tuple

# solving for circle using normal equations
def leastSquaresCircle(pts: np.array) -> Tuple[np.array, float]:
    # build linear system, see for example here https://math.stackexchange.com/questions/214661/circle-least-squares-fit
    A = np.stack(
        (2*pts[:, 0], 2*pts[:, 1], np.ones(pts.shape[0])), axis=1
    )
    b = np.stack(
        np.square(pts[:, 0]) + np.square(pts[:, 1])
    )

    # solve system
    x, res, rank, s = np.linalg.lstsq(A, b)

    # solve for radius, x2 = r^2 - x0^2 - x1^2
    r = np.sqrt(x[2] + x[0]**2 + x[1]**2)

    return x[:-1], r






def threePointCircle(p1: np.array, p2: np.array, p3: np.array) -> Tuple[np.array, float]:
    """Computes a circle, given 3 points on that circle, see https://stackoverflow.com/questions/26222525/opencv-detect-partial-circle-with-noise.

    Args:
        p1 (np.array): Point 1 on circle in OpenCV convention.
        p2 (np.array): Point 2 on circle in OpenCV convention.
        p3 (np.array): Point 3 on circle in OpenCV convention.

    Return:
        center (np.array): Center of circle
        radius (float): Radius of circle
    """
    x1, x2, x3 = p1[0], p2[0], p3[0]
    y1, y2, y3 = p1[1], p2[1], p3[1]

    center = np.array([
        ((x1*x1+y1*y1)*(y2-y3) + (x2*x2+y2*y2)*(y3-y1) + (x3*x3+y3*y3)*(y1-y2))/(2*(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2)),
        ((x1*x1+y1*y1)*(x3-x2) + (x2*x2+y2*y2)*(x1-x3) + (x3*x3+y3*y3)*(x2-x1))/(2*(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2))
    ])

    radius = np.sqrt((center[0]-x1)*(center[0]-x1) + (center[1]-y1)*(center[1]-y1))

    return center, radius


class EndoscopyCircleDetection():
    def __init__(buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []
        self.last_center = np.array([])
        self.last_radius = None

    def findCircle(th1: float=5. , th2: float=.5, th3: float=10., n_pts: int=100) -> Tuple[np.array, float]: # todo move to same unit th1 and 2
        # update buffer
        # build system
        # solve system
        # return
        pass

    def _buildLinearSystem(pts: np.array) -> Tuple[np.array, np.array]:
        """Build linear system that describes circle, for example check https://math.stackexchange.com/questions/214661/circle-least-squares-fit

        Args:

        Return:
            A (np.array): Linear system matrix
            b (np.array): Offset to linear equation
        """
        A = np.stack(
            (2*pts[:, 0], 2*pts[:, 1], np.ones(pts.shape[0])), axis=1
        )
        b = np.stack(
            np.square(pts[:, 0]) + np.square(pts[:, 1])
        )

        return A, b

    def _solveSystem(A, b) -> Tuple[np.array, float]:
        """Solve linear system for center and radius, for example check https://math.stackexchange.com/questions/214661/circle-least-squares-fit

        Args:

        Return:
            center (np.array): Circle's center
            radius (float): Circles radius
        """
        x, res, rank, s = np.linalg.lstsq(A, b)

        # solve for radius, x2 = r^2 - x0^2 - x1^2
        radius = np.sqrt(x[2] + x[0]**2 + x[1]**2)

        return x[:-1], radius

    # solving for circle using normal equations
    def leastSquaresCircle(pts: np.array) -> Tuple[np.array, float]:
        # build linear system, see for example here https://math.stackexchange.com/questions/214661/circle-least-squares-fit
        A = np.stack(
            (2*pts[:, 0], 2*pts[:, 1], np.ones(pts.shape[0])), axis=1
        )
        b = np.stack(
            np.square(pts[:, 0]) + np.square(pts[:, 1])
        )

        # solve system
        x, res, rank, s = np.linalg.lstsq(A, b)

        # solve for radius, x2 = r^2 - x0^2 - x1^2
        r = np.sqrt(x[2] + x[0]**2 + x[1]**2)

        return x[:-1], r


if __name__ == '__main__':
    import os

    # prefix = os.getcwd()
    prefix = '/home/martin/Documents/code_snippets/homography_imitation_learning/utils/processing'
    file = 'sample.mp4'

    vr = cv2.VideoCapture(os.path.join(prefix, file))

    # params
    N = 1         # moving average
    noise = 5     # first threshold for detection of black region in endoscope
    th2 = 0.5     # second threshold for moving average image
    n_pts = 100   # number of points sampled on segmented circle
    dist_th = 10  # outlier threshold for sampled points

    # to track
    avg_lst = []
    last_center = np.array([])
    last_radius = None
 
    while vr.isOpened():

        _, bgr = vr.read()
        if bgr is None:
            break
        bgr = cv2.resize(bgr, (640, 360))
        gra = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        gra = np.where(gra < noise, 0, 255).astype(np.uint8)

        # # removes arteries really well, might become part of homography regression data augmentation
        # pre = cv2.edgePreservingFilter(bgr)
        # can = cv2.Canny(gra, threshold1=200, threshold2= 500)  # sobel or canny
        can = cv2.Sobel(gra, cv2.CV_8U, 1, 1)

        avg_lst.append(can)

        if len(avg_lst) == N:
            avg = np.array(avg_lst)
            avg = avg.mean(axis=0)
            avg = avg.astype(np.float)/avg.max()
            avg = np.where(avg < th2, 0., 1.)  # possibly second thresholding
            
            avg_lst.pop(0)

            # get edges
            edges = np.where(avg > 0.5)

            # randomly sample points
            idcs = np.random.choice(np.arange(edges[0].size), size=n_pts,replace=False)
            pts = np.stack((edges[0][idcs], edges[1][idcs]), axis=1)

            # remove outliers
            if last_center.size is not 0 and last_radius:
                distance_to_center = np.linalg.norm(last_center - pts, axis=1)
                del_idx = np.where(np.abs(last_radius - distance_to_center) > dist_th)
                pts = np.delete(pts, del_idx, axis=0).reshape(-1, 2)

            # visualize edges
            for idc in idcs: 
                cv2.circle(avg, (edges[1][idc], edges[0][idc]), 6, (1,1,1))

            center, radius = leastSquaresCircle(pts)
            last_center, last_radius = center, radius

            center = center.astype(np.int)
            radius = int(radius)
            cv2.circle(avg, (center[1], center[0]), radius, (1,1,1))

            cv2.imshow('avg', avg)

        cv2.imshow('can', can)
        cv2.imshow('gra', gra)
        cv2.imshow('bgr', bgr)
        # cv2.imshow('pre', pre)
        cv2.waitKey()



