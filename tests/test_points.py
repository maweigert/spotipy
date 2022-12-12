import numpy as np
from spotipy.utils import points_to_prob
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from spotipy.utils import points_to_flow, cluster_flow, prob_to_points




if __name__ == '__main__':

    np.random.seed(42) 

    N = 128
    
    p0 = np.random.uniform(10,N-10,(N**2//100,2))

    # dp = .6
    # dp = 1.
    # p0 = np.array([[N/2, N/2-dp], [N/2, N/2+dp]]) + .1*np.random.uniform(-1,1,(2,2))
    

    u = points_to_flow(p0, (N,N), sigma=2)

    thresh = .3
    p1 = prob_to_points(u[...,0], prob_thresh=thresh)

    p2 = cluster_flow(u, u[...,0]>thresh, niter=50)[1]

    p = np.stack([cluster_flow(u, u[...,0]>thresh, niter=n, dt=.2)[1] for n in range(50)], 0)
    pp = np.concatenate([np.concatenate((i*np.ones((len(_p),1)), _p), -1) for i,_p in enumerate(p)], 0)


    # napshow(u[...,0])
    # napshow(p0, size=1, symbol='disc', face_color='g', edge_color='g', opacity=.8)
    # napshow(pp, size=1, symbol='disc', face_color='r', edge_color='r', opacity=.8)
    

    # y = points_to_prob(p, shape=(N,N), sigma=1) 
    # u = points_to_flow(p, shape=(N,N), sigma=2) 

    
