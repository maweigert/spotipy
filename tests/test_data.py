import numpy as np
from augmend import Augmend, Elastic


from spotipy.model import Config, SpotNetData, SpotNet
from spotipy.utils import points_to_prob, points_to_flow


if __name__ == '__main__':

    
    config = Config(axes = "YXC", n_channel_in=1)

    model = SpotNet(config, name = None, basedir = None)

    N = 200
    
    P = np.random.randint(10,N-10,(5,30,2))

    Y = np.stack(tuple(points_to_prob(p, shape=(N,N), sigma=1) for p in P))
    F = np.stack(tuple(points_to_flow(p, shape=(N,N), sigma=2) for p in P))

    X = Y + .1*np.random.normal(0,1,Y.shape)
    
        
    # aug = Augmend()
    # t = Elastic(amount=5, grid=5, order=0,axis = (0,1))
    # aug.add([t,t])

    data = SpotNetData(X,Y,F,patch_size=(128,128), length=1000, augmenter=None)


    
