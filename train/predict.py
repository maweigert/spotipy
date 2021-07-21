import numpy as np
from tifffile import imread
from tqdm import tqdm 
from csbdeep.utils import normalize
from pathlib import Path
import argparse
from spotipy.model import SpotNet
from spotipy.utils import points_to_prob
import matplotlib.pyplot as plt

np.random.seed(42)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training script for a spot net')

    parser.add_argument("-i","--input", type=str, nargs='+', 
                        help = "input images")
    parser.add_argument("-o","--output", type=str, default=None,
                        help = "output directory")
    parser.add_argument("-m", "--model", type=str,
                        default="models/train_multiscale_True_bce_aug_3_sigma_1.0_batch_4_n_101")

    args = parser.parse_args()



    model = SpotNet(None, args.model)

    plt.ion()
    plt.figure(figsize=(7,3.9))

    for f in args.input:
        print(f)
        x = imread(f)
        assert x.ndim==2
        
        x = normalize(x)
        
        n_tiles = tuple(max(1,s//2048) for s in x.shape)
        prob, points = model.predict(x, min_distance=1, n_tiles=n_tiles)#, prob_thresh=.5)


        plt.clf()
        plt.subplot(1,2,1)
        x_ = np.clip(x, 0,1)**.8
        plt.imshow(x_,cmap = "gray")
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(x_,cmap = "gray")
        plt.plot(*points.T[::-1],"o", color='C1', markerfacecolor='none')
        plt.axis('off')

        plt.suptitle(f'{Path(f).name}')
        plt.tight_layout()

        if args.output is not None:
            plt.savefig(f"{Path(args.output)}/result_{Path(f).name}.png")
