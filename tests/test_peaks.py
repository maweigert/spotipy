import numpy as np 
from spotipy.peaks import nms_points_2d, local_peaks

def demo_nms():
    np.random.seed(42) 

    p = np.random.uniform(0,100,(1000,2))
    # n = 10
    # Y, X = np.meshgrid(np.linspace(0,100,n), np.linspace(0,100,n), indexing="ij")
    # p = np.stack([Y.ravel(), X.ravel()], axis=1)

    min_dist = 10

    print(p)

    p2 = nms_points_2d(p, min_distance=min_dist)

    print(p2)

    import matplotlib.pyplot as plt
    plt.ion() 
    plt.figure(num=1)
    plt.clf()
    plt.plot(*p.T[::-1],'o')
    plt.plot(*p2.T[::-1],'o')
    plt.axis('equal')



def test_peaks():
    from skimage.feature import peak_local_max, corner_peaks
    import scipy.ndimage as ndi
    from timeit import default_timer as timer

    np.random.seed(42) 

    n = 2048
    x = np.zeros((n, n), np.float32)
    n_spots = 200*(n//256)**2
    x[tuple(np.random.randint(10,n-10, (2,n_spots)))] = np.random.uniform(.4,1,n_spots)
    x = ndi.gaussian_filter(x, 3)
    x /= x.max()

    # x [4:,4:]=  0*x[4:,4:]+1

    min_distance = 3

    t = timer()
    p1 = peak_local_max(x, min_distance=min_distance, threshold_abs=.1, threshold_rel=0)
    t = timer() - t
    print(f"peak_local_max: {t:.3f} s")

    t = timer()
    p2 = corner_peaks(x, min_distance=min_distance, threshold_abs=.1, threshold_rel=0)
    t = timer() - t
    print(f"corner_peaks: {t:.3f} s")

    t = timer()
    p3 = local_peaks(x, min_distance=min_distance, threshold_abs=.1, threshold_rel=0, exclude_border=False)
    t = timer() - t
    print(f"local : {t:.3f} s")
    


if __name__ == "__main__":
    # test_peaks()

    x = np.zeros((100,100))
    x[0,0] = 1
    
    p1 = local_peaks(x, min_distance=1, threshold_abs=.1, threshold_rel=0, exclude_border=False)
    p2 = local_peaks(x, min_distance=1, threshold_abs=.1, threshold_rel=0, exclude_border=True)
