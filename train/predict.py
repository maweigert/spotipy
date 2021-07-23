import numpy as np
from tifffile import imread
from tqdm import tqdm 
from csbdeep.utils import normalize
from csbdeep.utils.tf import limit_gpu_memory
limit_gpu_memory(1, total_memory=20000)
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
    parser.add_argument("-n","--norm", type=float, nargs=2, default=[1,99.8],
                        help = "normalization")
    parser.add_argument("-p","--plot", action="store_true")
    parser.add_argument("-m", "--model", type=str,
                        default="models/train_curated_multiscale_True_mae_aug_3_sigma_1.0_batch_4_n_215")

    args = parser.parse_args()


    model = SpotNet(None, args.model)

    if args.output is not None:
        Path(args.output).mkdir(exist_ok=True, parents=True)


    for f in args.input:
        print(f)
        x = imread(f)
        assert x.ndim==2
        
        x = normalize(x, *args.norm)
        
        n_tiles = tuple(max(1,s//4096) for s in x.shape)
        prob, points = model.predict(x, min_distance=1, n_tiles=n_tiles)

        if args.plot:
            plt.ion()
            plt.figure(figsize=(10,3.9))
        
            plt.clf()
            plt.subplot(1,3,1)
            x_ = np.clip(x, 0,1)**.7
            plt.imshow(x_,cmap = "gray")
            plt.axis('off')

            plt.subplot(1,3,2)
            plt.imshow(x_,cmap = "gray")
            plt.plot(*points.T[::-1],"o", color='C1', markerfacecolor='none')
            plt.axis('off')

            plt.subplot(1,3,3)
            plt.imshow(prob)
            plt.axis('off')

            plt.suptitle(f'{Path(f).name}')
            plt.tight_layout()

            if args.output is not None:
                plt.savefig(f"{Path(args.output)}/result_{Path(f).name}.png")

        if args.output is not None:
            np.savetxt(f"{Path(args.output)}/result_norm_{args.norm[0]:.1f}_{args.norm[1]:.1f}_{Path(f).name}.csv", points.astype(int), delimiter=",")
