import numpy as np
from spotipy.utils import points_to_prob
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from spotipy.utils import points_to_flow
from spotipy.lib.spotflow2d import c_spotflow2d

def points_to_flow(points, shape, sigma = 3):
    x = np.zeros(shape, np.float32)
    points = np.asarray(points).astype(np.int32)
    assert points.ndim==2 and points.shape[1]==2
    D = cdist(points, points)
    A = D < 8*sigma+1
    np.fill_diagonal(A,False) 
    G = nx.from_numpy_array(A)
    
    mask = np.zeros(shape, np.float32)
    flow_x = np.zeros(shape, np.float32)
    flow_y = np.zeros(shape, np.float32)
    
    while len(G)>0:
        inds = nx.maximal_independent_set(G)
        gauss = np.zeros(shape, np.float32)
        gauss[tuple(points[inds].T)] = 1
        g = gaussian_filter(gauss, sigma, mode=  "constant")
        g /= (np.max(g)+1e-10)
        fy = gaussian_filter(gauss, sigma, order=(1,0), mode=  "constant")
        fx = gaussian_filter(gauss, sigma, order=(0,1), mode=  "constant")
        fx /= (np.max(fx)+1e-10)
        fy /= (np.max(fy)+1e-10)
        m = g>mask 
        flow_y[m] = fy[m]
        flow_x[m] = fx[m]
        mask = np.maximum(mask,g)
        G.remove_nodes_from(inds)
    flow = np.stack([flow_y,flow_x],-1)
    return flow



if __name__ == '__main__':

    np.random.seed(42) 

    N = 128
    
    p = np.random.uniform(10,N-10,(N**2//100,2))

    u = points_to_flow(p, (N,N), sigma=1)

    # y = points_to_prob(p, shape=(N,N), sigma=1) 
    # u = points_to_flow(p, shape=(N,N), sigma=2) 

    
