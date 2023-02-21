import numpy as np 
import scipy.ndimage as ndi
from timeit import default_timer as timer
from spotipy.peaks import maximum_filter_2d 

def test_maximum_filter():

    for kernel_size in (1,3,5):
        for shape in ((11,16), (256,256), (1024,1024)):

            print(f"kernel_size: {kernel_size}, shape: {shape}")
            x = np.random.uniform(0,100, shape).astype(np.float32)

            t = timer()
            u1 = ndi.maximum_filter(x, kernel_size, mode='constant')
            t = timer() - t
            print(f"scipy:  {t:.4f} s")

            t = timer()
            u2 = maximum_filter_2d(x, np.int32(kernel_size))
            t = timer() - t
            print(f"openmp: {t:.4f} s")

            print(f'difference {np.abs(u1-u2).max():.4f}')
            assert np.allclose(u1, u2)

if __name__ == "__main__":
    
    test_maximum_filter()

    