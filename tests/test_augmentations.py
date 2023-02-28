import numpy as np 
from scipy import ndimage as ndi
from spotipy.utils import points_to_prob
from spotipy.augmentations import FlipRot90, Rotate, AugmentationPipeline, IntensityScaleShift

def dummy_data():
    x = np.random.uniform(0,1,(64, 64))
    p = np.random.randint(0,64,(21, 2))
    x[tuple(p.T.tolist())] = np.random.uniform(5,10,len(p))
    x = ndi.gaussian_filter(x, 1)
    return x, p


if __name__ == "__main__":
    x, p = dummy_data()

    import matplotlib.pyplot as plt

    plt.ion() 

    aug = AugmentationPipeline() 
    aug.add(FlipRot90(), .8)
    aug.add(Rotate(), .8)
    aug.add(IntensityScaleShift(), .8)

    n = 5

    for i in range(n):
        x2, p2 = aug(x,p)
        print(i, p2.max())
            
        plt.subplot(1, n, i+1)
        plt.cla()
        plt.imshow(x2, clim=(0,1))
        plt.plot(*p2.T[::-1], 'C1.')

    # ts = (FlipRot90(), Rotate())    
    # for i, t in enumerate(ts):
    #     for j in range(n):
            
    #         x2, p2 = t(x,p)
    #         plt.subplot(len(ts), n, i*n +j +1)
    #         plt.cla()
    #         plt.imshow(x2)
    #         plt.plot(*p2.T[::-1], 'C1.')
