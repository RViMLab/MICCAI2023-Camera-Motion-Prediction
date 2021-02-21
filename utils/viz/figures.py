import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def warp_figure(img: np.array, uv: np.array, duv: np.array, duv_pred: np.array, H: np.array) -> plt.Figure:
    r"""

    Args:
        img (np.array): Original RGB image of shape HxWxC [0 ... 1]
        uv (np.array): Crop edges of shape 4x2
        duv (np.array): Deviation from crop edges uv of shape 4x2 
        duv_pred (np.array): Predicted deviation from crop edges uv of shape 4x2
        H (np.array): Homography corresponding to duv of shape 3x3

    Return:
        figure (plt.Figure): Matplotlib figure
    """
    figure = plt.figure()

    # Apply homography 
    wrp_uv = uv + duv
    wrp_uv_pred = uv + duv_pred

    wrp = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))

    # Visualize warped uv, and predicted uv
    gt_col, pred_col = (0, 1, 1), (1, 1, 0)
    cv2.polylines(wrp, [wrp_uv[:,::-1].astype(np.int32)], isClosed=True, color=gt_col, thickness=2)
    cv2.polylines(wrp, [wrp_uv_pred[:,::-1].astype(np.int32)], isClosed=True, color=pred_col, thickness=2)

    ax = figure.add_subplot(111)
    gt_patch = mpatches.Patch(color=gt_col, label='Ground Truth')
    pred_patch = mpatches.Patch(color=pred_col, label='Prediction')
    ax.legend(handles=[gt_patch, pred_patch])
    plt.imshow(wrp)

    return figure


def duv_mean_pairwise_distance_figure(duvs: np.ndarray, re_fps: int, fps: int, dpi: int=200) -> plt.Figure:
    r"""Helper function to plot the severity of a motion described by a homography.

    Args:
        duvs (np.ndarray): Edge delta of shape Nx4x2
        re_fps (int): Frame rate of re-sampled video
        fps (int): Frame rate of initial video
        dpi (int): Figure resolution

    Return:
        figure (plt.Figure): Matplotlib figure that shows homography norm over the course of the sequence
    """
    figure = plt.figure(dpi=200)

    # Mean pairwise distance
    duvs_mpd = np.linalg.norm(duvs, axis=2).mean(axis=1)

    plt.title('Camera Motion')
    plt.plot(
        np.arange(start=0, stop=re_fps*duvs_mpd.size, step=re_fps)/fps, 
        duvs_mpd, 
        label='Resampled frame rate {} at {} fps'.format(re_fps, int(fps))
    )
    plt.xlabel('Time / s')
    plt.ylabel('Mean Pairwise Distance / pixel')
    plt.grid()
    plt.legend()

    return figure

