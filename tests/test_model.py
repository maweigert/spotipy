import numpy as np 
from spotipy.model import Config, SpotNetData, SpotNet
from spotipy.utils import points_to_prob


if __name__ == '__main__':

    
    config = Config(axes = "YXC", n_channel_in=1)

    model = SpotNet(config, name = None, basedir = None)

    X = np.zeros((5,128,128))

    P = np.random.randint(0,128,(5,10,2))

    Y = np.stack(tuple(points_to_prob(p, shape=X.shape[1:], sigma=1) for p in P))
    
    model.train(X,Y, validation_data=[X,Y], steps_per_epoch=10)



    
