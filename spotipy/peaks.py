import numpy as np
import scipy.ndimage as ndi
from skimage.feature.peak import _get_excluded_border_width, _get_threshold, _exclude_border
import os

def get_num_threads():
    # set OMP_NUM_THREADS to 1/2 of the number of CPUs by default
    n_cpu = os.cpu_count()
    n_threads = int(os.environ.get("OMP_NUM_THREADS",n_cpu))
    n_threads = max(1,min(n_threads, n_cpu//2))
    return n_threads



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
    from spotipy.lib.point_nms import c_point_nms_2d

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



def maximum_filter_2d(image:np.ndarray, kernel_size:int=3) -> np.ndarray:
    from spotipy.lib.filters import c_maximum_filter_2d_float

    if not image.ndim==2:
        raise ValueError("Image must be 2D")
    if not kernel_size>0 and kernel_size%2==1:
        raise ValueError("kernel_size must be positive and odd")

    image = np.ascontiguousarray(image, dtype=np.float32)
    n_threads = get_num_threads()
    return c_maximum_filter_2d_float(image, np.int32(kernel_size//2), np.int32(n_threads))


def local_peaks(image:np.ndarray, min_distance=1, exclude_border=True, threshold_abs=None, threshold_rel=None):
    if not image.ndim==2 and not image.ndim==2:
        raise ValueError("Image must be 2D")

    # make compatible with scikit-image 
    # https://github.com/scikit-image/scikit-image/blob/a4e533ea2a1947f13b88219e5f2c5931ab092413/skimage/feature/peak.py#L120
    border_width = _get_excluded_border_width(image, min_distance, exclude_border)
    threshold = _get_threshold(image, threshold_abs, threshold_rel)

    image = image.astype(np.float32)

    if min_distance<=0:
        mask = image > threshold
    else:
        mask = maximum_filter_2d(image, 2*min_distance+1) == image
        
        # no peak for a trivial image
        image_is_trivial = np.all(mask) 
        if image_is_trivial:
            mask[:] = False
        mask &= image > threshold

    mask = _exclude_border(mask, border_width)    

    coord = np.nonzero(mask)
    intensities = image[coord]
    coord = np.stack(coord, axis=1)
    points = (nms_points_2d(coord, min_distance=min_distance)).astype(int)

    return points
