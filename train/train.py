import numpy as np
from tifffile import imread
from tqdm import tqdm 
from csbdeep.utils import normalize
from csbdeep.utils.tf import limit_gpu_memory
limit_gpu_memory(1, total_memory=6000)

from pathlib import Path
from augmend import Augmend, Elastic, Identity, FlipRot90, AdditiveNoise, Scale, GaussianBlur, Rotate, IntensityScaleShift, DropEdgePlanes
import argparse
from sklearn.model_selection import train_test_split
from spotipy.model import Config, SpotNetData, SpotNet
from spotipy.utils import points_to_prob, str2bool

np.random.seed(42)


def get_data(folder, nfiles= None, sigma= 1):
    """points will be in (x,y) order! (reversed than for model.predict!) """
   
    folder = Path("data")/folder
    
    fx = sorted(folder.glob("*.tif"))

    fy = sorted(folder.glob(f"*.npy"))
  
    if not len(fx)>0:
        raise ValueError("empty folder!")
    
    if not len(fx)==len(fy):
        raise ValueError("unequal number of images and csv annotations!")
    
    assert len(fx) == len(set(fx))

    fx = fx[:nfiles]
    fy = fy[:nfiles]

    X = tuple(normalize(imread(f)) for f in tqdm(fx))

    P = tuple(np.load(f)[:,::-1] for f in tqdm(fy))
    

    Y = tuple(points_to_prob(p, x.shape[:2], sigma = sigma, mode = "max") for p,x in tqdm(zip(P,X), total=len(X)))

    X,Y = np.stack(X), np.stack(Y)          

    return X, Y, P

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training script for a spot net')

    parser.add_argument("-n","--epochs", type=int, default=200,
                        help = "number of epochs to train")
    parser.add_argument("-o","--output", type=str, default="models")
    parser.add_argument("-d", "--dataset", type=str,
                        default="train_curated")
    parser.add_argument("-s", "--sigma", type=float, default=1.,
                        help = "sigma for gaussian blobs")
    parser.add_argument("--nfiles", type=int, default =None)
    parser.add_argument("--dry", action='store_true',
                            help = "dry run (no outputs are generated)")
    parser.add_argument("--augment", type=int, default=1,
                        help = "augmentation level (0,1,2)")
    parser.add_argument("--batch_size", type=int, default=4),
    parser.add_argument("--steps_per_epoch", type=int, default=1024),
    parser.add_argument("--loss", type=str, choices = ["bce","scale_sum","mae", "mse","focal"], default = "bce")
    parser.add_argument("--multiscale", type=str2bool, default="y"),

    args = parser.parse_args()



    X, Y, P = get_data(folder=args.dataset, sigma=args.sigma, nfiles=args.nfiles)
    X, Xv, Y, Yv, P, Pv = train_test_split(X,Y,P, test_size=max(1, len(X)//12), random_state=37)


    print(f'number of training   images:  {len(X)}')
    print(f'number of validation images:  {len(Xv)}')
    config = Config(n_channel_in=1,
                    unet_n_depth = 3,
                    unet_pool = 2,
                    spot_weight = 0 if args.loss in ("scale_sum", "focal") else 10,
                    multiscale = args.multiscale,
                    mode = args.loss,
                    train_learning_rate = 3e-4, 
                    spot_weight_decay = .0,
                    train_batch_size=args.batch_size)

    name = f"{args.dataset}_multiscale_{args.multiscale}_{args.loss}_aug_{args.augment}_sigma_{args.sigma:.1f}_batch_{args.batch_size}_n_{args.epochs}"
    model = SpotNet(config, name = None if args.dry else name, basedir = None if args.dry else args.output)


    
    if args.augment==0:
        aug = None

    elif args.augment==1:
        aug = Augmend()
        aug.add([FlipRot90(axis = (0,1)),FlipRot90(axis = (0,1))])
        
    elif args.augment==2:
        aug = Augmend()
        aug.add([FlipRot90(axis = (0,1)),FlipRot90(axis = (0,1))])
        aug.add([Rotate(axis = (0,1)),Rotate(axis = (0,1))])
        aug.add([AdditiveNoise(sigma=(0,.04)),Identity()], probability=.5)
        aug.add([IntensityScaleShift(axis = (0,1), scale=(.8,1.3), shift=(-.1,.1)),Identity()])
        
    elif args.augment==3:
        aug = Augmend()
        aug.add([FlipRot90(axis = (0,1)),FlipRot90(axis = (0,1))])
        aug.add([Rotate(axis = (0,1)),Rotate(axis = (0,1))])
        t = Elastic(amount=5, grid=10, order=0,axis = (0,1))
        aug.add([t,t])
        aug.add([GaussianBlur(axis = (0,1),amount = (0,1.5)),Identity()], probability=.5)
        aug.add([AdditiveNoise(sigma=(0,.04)),Identity()], probability=.5)
        aug.add([IntensityScaleShift(axis = (0,1), scale=(.5,2.), shift=(-.1,.1)),Identity()])
        aug.add([DropEdgePlanes(width=16), DropEdgePlanes(width=16)], probability=.2)

    else:
        raise NotImplementedError(args.augment)


    
    if args.epochs>0:
        model.train(X,Y, validation_data = [Xv,Yv],
                    augmenter = aug,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch//args.batch_size,
                        workers=args.batch_size)
            
        model.optimize_thresholds(Xv, Yv, verbose=2, save_to_json=False if args.dry else True, optimize_kwargs=dict(bracket=(.3,.7)))
                               
