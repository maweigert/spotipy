import numpy as np
import scipy.ndimage as ndi
from skimage.feature.peak import _get_excluded_border_width, _get_threshold, _exclude_border
from spotipy.lib.point_nms import c_point_nms_2d


def nms_points_2d(points: np.ndarray, scores: np.ndarray = None, min_distance:int=2) -> np.ndarray:
    """Non-maximum suppression for 2D points, choosing the highest scoring points while 
    ensuring that no two points are closer than min_distance.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N,2) containing the points to be filtered.
    scores : np.ndarray
        Array of shape (N,) containing scores for each point
        If None, all points have the same score 
    min_distance : int, optional
        Minimum distance between points, by default 2

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing the indices of the points that survived the filtering.
    """
    points = np.asarray(points)
    if not points.ndim == 2 and points.shape[1] == 2:
        raise ValueError("points must be a array of shape (N,2)")
    if scores is None: 
        scores = np.ones(len(points))
    else:
        scores = np.asarray(scores)
    if not scores.ndim == 1:
        raise ValueError("scores must be a array of shape (N,)")

    idx = np.argsort(scores, kind='stable')
    points = points[idx]
    scores = scores[idx]

    points = np.ascontiguousarray(points, dtype=np.float32)
    inds = c_point_nms_2d(points, np.float32(min_distance))
    return points[inds].copy()



def local_peaks(image:np.ndarray, min_distance=1, exclude_border=True, threshold_abs=None, threshold_rel=None):
    if not image.ndim==2 and not image.ndim==2:
        raise ValueError("Image must be 2D")

    # make compatible with scikit-image 
    # https://github.com/scikit-image/scikit-image/blob/a4e533ea2a1947f13b88219e5f2c5931ab092413/skimage/feature/peak.py#L120
    border_width = _get_excluded_border_width(image, min_distance, exclude_border)
    threshold = _get_threshold(image, threshold_abs, threshold_rel)

    #    mask = _get_peak_mask(image, footprint, threshold)
    if min_distance<=0:
        mask = image > threshold
    else:
        mask = ndi.maximum_filter(image, 2*min_distance+1, mode='nearest') == image
        
        # no peak for a trivial image
        image_is_trivial = np.all(mask) 
        if image_is_trivial:
            mask[:] = False
        mask &= image > threshold

    mask = _exclude_border(mask, border_width)    

    coord = np.nonzero(mask)
    intensities = image[coord]
    coord = np.stack(coord, axis=1)

    points = nms_points_2d(coord, min_distance=min_distance)
    return points
