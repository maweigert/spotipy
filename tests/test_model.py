import numpy as np 
from spotipy.model import Config, SpotNetData, SpotNet
from spotipy.utils import points_to_prob


def dummy_data(n_samples=16):
    X = np.random.uniform(0,1,(n_samples, 128, 128))
    P = np.random.randint(0,128,(n_samples, 21, 2))
    for x, p in zip(X, P):
        x[tuple(p.T.tolist())] = np.random.uniform(2,5,len(p))
    Y = np.stack(tuple(points_to_prob(p, (128,128)) for p in P))
    return X, Y, P

if __name__ == '__main__':

    
    config = Config(n_channel_in=1, train_patch_size=(64,64), backbone='unet', spot_sigma=1)

    model = SpotNet(config, name = 'test', basedir = 'models')

    X,Y, P = dummy_data(128)
    Xv,Yv, Pv = dummy_data(16)

    model.train(X,P, validation_data=[Xv, Pv], epochs=10, steps_per_epoch=128)

    # model.optimize_thresholds(Xv,Pv)

    # points, _ = model.predict(X[0])

    
