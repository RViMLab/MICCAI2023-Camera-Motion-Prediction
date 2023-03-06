import numpy as np


def image_coordinate_to_angle(
    duv_i_0: float, duv_i_1: float, duv_i_length: float
) -> float:
    r"""Computes the angle between duv vector and the x axis in the image coordinate system.

    Args:
        duv_i_0 (float): y component of the duv vector
        duv_i_1 (float): x component of the duv vector
        duv_i_length (float): length of the duv vector

    Returns:
        ang (float): angle between duv vector and the x axis in the image coordinate system
    """
    ang = np.arccos(duv_i_1 / duv_i_length)
    if -duv_i_0 >= 0:
        return ang
    if -duv_i_0 < 0:
        return 2 * np.pi - ang


def angle_to_translate(
    duv_0_0_ang: float, duv_1_0_ang: float, duv_2_0_ang: float, duv_3_0_ang: float
) -> str:
    r"""Classifies the angle between the duv vector and the x axis in the image coordinate system
    to a translation type.

    Args:
        duv_0_0_ang (float): Top left duv vector angle
        duv_1_0_ang (float): Top right duv vector angle
        duv_2_0_ang (float): Bottom right duv vector angle
        duv_3_0_ang (float): Bottom left duv vector angle

    Returns:
        str: translation type [right, left, up, down]
    """
    duv_0_quarter = np.floor(duv_0_0_ang / (np.pi / 2))
    duv_1_quarter = np.floor(duv_1_0_ang / (np.pi / 2))
    duv_2_quarter = np.floor(duv_2_0_ang / (np.pi / 2))
    duv_3_quarter = np.floor(duv_3_0_ang / (np.pi / 2))

    if (
        (duv_0_quarter == 0 or duv_0_quarter == 3) and
        (duv_1_quarter == 0 or duv_1_quarter == 3) and
        (duv_2_quarter == 0 or duv_2_quarter == 3) and
        (duv_3_quarter == 0 or duv_3_quarter == 3)
    ):
        return "right"
    if (
        (duv_0_quarter == 1 or duv_0_quarter == 2) and
        (duv_1_quarter == 1 or duv_1_quarter == 2) and
        (duv_2_quarter == 1 or duv_2_quarter == 2) and
        (duv_3_quarter == 1 or duv_3_quarter == 2)
    ):
        return "left"
    if (
        (duv_0_quarter == 0 or duv_0_quarter == 1) and
        (duv_1_quarter == 0 or duv_1_quarter == 1) and
        (duv_2_quarter == 0 or duv_2_quarter == 1) and
        (duv_3_quarter == 0 or duv_3_quarter == 1)
    ):
        return "up"
    if (
        (duv_0_quarter == 2 or duv_0_quarter == 3) and
        (duv_1_quarter == 2 or duv_1_quarter == 3) and
        (duv_2_quarter == 2 or duv_2_quarter == 3) and
        (duv_3_quarter == 2 or duv_3_quarter == 3)
    ):
        return "down"
    return "mixture"


def angle_to_zoom_rotate(
    duv_0_0_ang: float, duv_1_0_ang: float, duv_2_0_ang: float, duv_3_0_ang: float
) -> str:
    r"""Classifies the angle between the duv vector and the x axis in the image coordinate system
    to a zoom or rotate type.

    Args:
        duv_0_0_ang (float): Top left duv vector angle
        duv_1_0_ang (float): Top right duv vector angle
        duv_2_0_ang (float): Bottom right duv vector angle
        duv_3_0_ang (float): Bottom left duv vector angle

    Returns:
        str: zoom or rotate type [zoom_in, zoom_out, rotate_left, rotate_right]
    """
    duv_0_quarter = np.floor(duv_0_0_ang / (np.pi / 2))
    duv_1_quarter = np.floor(duv_1_0_ang / (np.pi / 2))
    duv_2_quarter = np.floor(duv_2_0_ang / (np.pi / 2))
    duv_3_quarter = np.floor(duv_3_0_ang / (np.pi / 2))

    if (
        duv_0_quarter == 1
        and duv_1_quarter == 0
        and duv_2_quarter == 3
        and duv_3_quarter == 2
    ):
        return "zoom_out"
    if (
        duv_0_quarter == 3
        and duv_1_quarter == 2
        and duv_2_quarter == 1
        and duv_3_quarter == 0
    ):
        return "zoom_in"
    if (
        duv_0_quarter == 0
        and duv_1_quarter == 3
        and duv_2_quarter == 2
        and duv_3_quarter == 1
    ):
        return "rotate_right"
    if (
        duv_0_quarter == 2
        and duv_1_quarter == 1
        and duv_2_quarter == 0
        and duv_3_quarter == 3
    ):
        return "rotate_left"
    return "mixture"


def classify_duv_motion(
    duv_0_0: float,
    duv_0_1: float,
    duv_1_0: float,
    duv_1_1: float,
    duv_2_0: float,
    duv_2_1: float,
    duv_3_0: float,
    duv_3_1: float,
    motion_threadhold: float = 10,
) -> str:
    r"""Classifies the motion type based on the duv vectors.

    Args:
        duv_0_0 (float): Top left duv vector y component
        duv_0_1 (float): Top left duv vector x component
        duv_1_0 (float): Top right duv vector y component
        duv_1_1 (float): Top right duv vector x component
        duv_2_0 (float): Bottom right duv vector y component
        duv_2_1 (float): Bottom right duv vector x component
        duv_3_0 (float): Bottom left duv vector y component
        duv_3_1 (float): Bottom left duv vector x component
        motion_threadhold (float, optional): Threshold for motion. Defaults to 10.

    Returns:
        str: motion type [static, mixture, left, right, up, down, zoom_in, zoom_out, rotate_left, rotate_right]
    """
    duv_0_length = np.sqrt(duv_0_0**2 + duv_0_1**2)
    duv_1_length = np.sqrt(duv_1_0**2 + duv_1_1**2)
    duv_2_length = np.sqrt(duv_2_0**2 + duv_2_1**2)
    duv_3_length = np.sqrt(duv_3_0**2 + duv_3_1**2)

    # amplitude below threshold is considered static
    if (
        duv_0_length < motion_threadhold
        and duv_1_length < motion_threadhold
        and duv_2_length < motion_threadhold
        and duv_3_length < motion_threadhold
    ):
        return "static"

    # compute angle between 0 unit vector
    duv_0_0_ang = image_coordinate_to_angle(duv_0_0, duv_0_1, duv_0_length)
    duv_1_0_ang = image_coordinate_to_angle(duv_1_0, duv_1_1, duv_1_length)
    duv_2_0_ang = image_coordinate_to_angle(duv_2_0, duv_2_1, duv_2_length)
    duv_3_0_ang = image_coordinate_to_angle(duv_3_0, duv_3_1, duv_3_length)

    # classify
    case = angle_to_translate(duv_0_0_ang, duv_1_0_ang, duv_2_0_ang, duv_3_0_ang)
    if case != "mixture":
        return case
    return angle_to_zoom_rotate(duv_0_0_ang, duv_1_0_ang, duv_2_0_ang, duv_3_0_ang)
