import numpy as np
from tifffile import imread
from tqdm import tqdm 
from csbdeep.utils import normalize
from csbdeep.utils.tf import limit_gpu_memory
limit_gpu_memory(1, total_memory=6000)

from pathlib import Path
import argparse
from spotipy.model import SpotNet
from spotipy.utils import points_to_prob, points_bipartite_matching
from sklearn.model_selection import train_test_split
from train import get_data 
import matplotlib.pyplot as plt

np.random.seed(42)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training script for a spot net')

    parser.add_argument("-n", type=int, nargs="+", default=[0])
    parser.add_argument("-m", "--model", type=str,
                        default="models/train_multiscale_True_mae_aug_3_sigma_1.0_batch_4_n_206")
    parser.add_argument("-o","--output", type=str, default=None,
                        help = "output directory")


    args = parser.parse_args()


    X, Y, P = get_data(folder="train_curated", sigma=1)
    X, Xv, Y, Yv, P, Pv = train_test_split(X,Y,P, test_size=max(1, len(X)//12), random_state=37)


    if args.output is not None:
        Path(args.output).mkdir(exist_ok=True, parents=True)
    

    model = SpotNet(None, args.model)


    for n in args.n:
        x, y, points_gt = Xv[n], Yv[n], Pv[n][:,::-1]
        x = normalize(x)
            
        n_tiles = tuple(max(1,s//4096) for s in x.shape)
        prob, points = model.predict(x, min_distance=1, n_tiles=n_tiles, prob_thresh=.4)

        stats = points_bipartite_matching(points_gt, points)

        plt.ion()
        plt.figure(figsize=(10,3.9))

        plt.clf()
        plt.subplot(1,3,1)
        x_ = np.clip(x, 0,1)**.7
        plt.imshow(x_,cmap = "gray")
        plt.plot(*points_gt.T[::-1],"o", color='magenta', markerfacecolor='none')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(x_,cmap = "gray")

        plt.plot(*points.T[::-1],"o", color='C2', markerfacecolor='none')
        if stats.fn>0:
            plt.plot(*points_gt[np.array(stats.false_negatives)].T[::-1],"o", color='magenta', markerfacecolor='none')
        if stats.fp>0:
            plt.plot(*points[np.array(stats.false_positives)].T[::-1],"o", color='darkred', markerfacecolor='none')

        plt.title("Prediction")
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(prob)
        plt.axis('off')

        plt.suptitle(f"accuracy = {stats.accuracy:.2f}")

        plt.tight_layout()

        if args.output is not None:
            plt.savefig(f"{Path(args.output)}/result_{n}.png")


