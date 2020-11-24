import numpy as np

def yt_alpha_blend(img_y: np.array, img_t: np.array, alpha: float=.5):
    r"""Blends RGB image into yellow and turquoise.

    Args:
        img_y (np.array): Image to be blended in yellow color
        img_t (np.array): Image to be blended in turquoise color

    Returns:
        blend (np.array): Blend of the form alpha*img_y + (1-alpha)*img_t
    """
    img_y_cpy = img_y.copy() 
    img_t_cpy = img_t.copy()

    img_y_cpy[...,0]  = 0
    img_t_cpy[...,-1] = 0

    blend = alpha*img_y_cpy + (1-alpha)*img_t_cpy
    return blend
