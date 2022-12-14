import numpy as np
from spotipy.utils import points_to_prob
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from spotipy.utils import points_to_flow, cluster_flow, prob_to_points, points_matching




if __name__ == '__main__':

    np.random.seed(42) 

    N = 128
    
    p0 = np.random.uniform(10,N-10,(N**2//40,2))

    # p0 = np.random.uniform(10,N-10,(2,2))

    # dp = .6
    # dp = 1.
    # p0 = np.array([[N/2, N/2-dp], [N/2, N/2+dp]]) + .1*np.random.uniform(-1,1,(2,2))
    # p0 = np.array([[N/2-dp, N/2], [N/2+dp, N/2]]) + .1*np.random.uniform(-1,1,(2,2))
    

    u = points_to_flow(p0, (N,N), sigma=1)

    thresh = .3
    p1 = prob_to_points(u[...,0], prob_thresh=thresh, min_distance=1)

    # p2, p2_full = cluster_flow(u, u[...,0]>thresh, niter=40, dt=.1)
    p2, p2_full = cluster_flow(u, u[...,0]>thresh, niter=300, dt=.1, atol=0.2)


    s1 = points_matching(p0,p1,cutoff_distance=1)
    s2 = points_matching(p0,p2,cutoff_distance=1)

    print(f'{s1.f1:.4f}') 
    print(f'{s2.f1:.4f}')

    p = np.stack([cluster_flow(u, u[...,0]>thresh, niter=n, dt=.1, atol=1e-3)[1] for n in range(100)], 0)
    pp = np.concatenate([np.concatenate((i*np.ones((len(_p),1)), _p), -1) for i,_p in enumerate(p)], 0)

# napclf()
# napshow(u[...,0])
# napshow(p0, size=1, symbol='disc', face_color='g', edge_color='g', opacity=.8)
# napshow(p1, size=1, symbol='disc', face_color='b', edge_color='b', opacity=.8)
# napshow(p2, size=1, symbol='disc', face_color='magenta', edge_color='magenta', opacity=.8)
# napshow(pp, size=.7, symbol='disc', face_color='r', edge_color='r', opacity=.5)
    
