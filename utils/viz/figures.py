import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def warp_figure(img: np.array, uv: np.array, duv: np.array, duv_pred: np.array, H: np.array):
    r"""

    Args:
        img (np.array): Original RGB image of shape HxWxC
        uv (np.array): Crop edges of shape 4x2
        duv (np.array): Deviation from crop edges uv of shape 4x2 
        duv_pred (np.array): Predicted deviation from crop edges uv of shape 4x2
        H (np.array): Homography corresponding to duv of shape 3x3

    Return:
        figure (plt.figure): Matplotlib figure
    """
    figure = plt.figure()

    # Apply homography 
    wrp_uv = uv + duv
    wrp_uv_pred = uv + duv_pred

    wrp = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))

    # Visualize warped uv, and predicted uv
    gt_col, pred_col = (0, 255, 255), (255, 255, 0)
    cv2.polylines(wrp, [wrp_uv[:,::-1].astype(np.int32)], isClosed=True, color=gt_col, thickness=2)
    cv2.polylines(wrp, [wrp_uv_pred[:,::-1].astype(np.int32)], isClosed=True, color=pred_col, thickness=2)

    ax = figure.add_subplot(111)
    gt_patch = mpatches.Patch(color=tuple(t/255 for t in gt_col), label='Ground Truth')
    pred_patch = mpatches.Patch(color=tuple(t/255 for t in pred_col), label='Prediction')
    ax.legend(handles=[gt_patch, pred_patch])
    plt.imshow(wrp)

    return figure
