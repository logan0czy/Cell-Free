import math
import numpy as np




def angle(tgt_loc, src_loc, src_loc_delta):
    """
    Compute the azimuth and elevation angle from source to target.
    If the source is planar, assume it is placed vertically.
    Parameters:
    tgt_loc (np.array): transmission target location, 3-D
    src_loc (np.array): transmission source location, 3-D
    src_loc_delta (np.array): shift from source

    Returns:
    theta_a (float): azimuth angle
    theta_e (float): elevation angle
    """
    normal_vec = np.zeros(3)
    if src_loc_delta[0] == 0:
        normal_vec[0] = 1
    elif src_loc_delta[1] == 0:
        normal_vec[1] = 1
    else:
        normal_vec[1] = 1
        normal_vec[0] = - normal_vec[1]*src_loc_delta[1]/src_loc_delta[0]
    proj_vec = np.array([tgt_loc[0], tgt_loc[1], src_loc[2]])

    theta_e = math.acos(np.linalg.norm(proj_vec-src_loc) / np.linalg.norm(tgt_loc-src_loc))
    inner_prod = abs(np.matmul(proj_vec-src_loc, normal_vec))
    theta_a = math.acos(inner_prod / (np.linalg.norm(proj_vec-src_loc)*np.linalg.norm(normal_vec)))
    return theta_a, theta_e