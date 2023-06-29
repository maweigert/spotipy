import numpy as np 
from spotipy.model import Config, SpotNetData, SpotNet
from spotipy.utils import points_to_prob
from spotipy.utils import normalize

np.random.seed(42) 

def dummy_data(n_samples=16):
    
    P = np.random.randint(0,256,(n_samples, 21, 2))
    X = np.stack(tuple(points_to_prob(p, (256,256), sigma=1.5) for p in P)) 
    Y = np.stack(tuple(points_to_prob(p, (256,256), sigma=1.5) for p in P))
    X = X + np.random.normal(0,0.1, X.shape)
    return X, Y, P


def test_train():

    config = Config(n_channel_in=1, train_patch_size=(64,64), backbone='unet', spot_sigma=1)

    model = SpotNet(config, name = None, basedir = None)

    X,Y, P = dummy_data(128)
    Xv,Yv, Pv = dummy_data(16)

    model.train(X,P, validation_data=[Xv, Pv], epochs=2, steps_per_epoch=128)

    model.optimize_thresholds(Xv,Pv)

    points, _ = model.predict(X[0])
    points2, _ = model.predict(X[0], peak_mode='fast')

if __name__ == '__main__':

    X,Y, P = dummy_data(128)
    
    
    model = SpotNet.from_pretrained('hybiss')
    
    
    p1, d1 = model.predict(X[0], scale=.73, return_details=True)
    p2, d2 = model.predict(X[0], scale=0.73, n_tiles=(2,2), return_details=True)
