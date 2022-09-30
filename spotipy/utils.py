import numpy as np
import sys
import datetime
import warnings
from csbdeep.utils import normalize
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import map_coordinates, zoom
from scipy.optimize import minimize_scalar
from skimage.measure import regionprops, label
from skimage.feature import corner_peaks, corner_subpix
from stardist.matching import matching
from skimage.draw import disk
from collections import namedtuple
from tqdm import tqdm
import networkx as nx
from scipy.spatial.distance import cdist
from types import SimpleNamespace

def _filter_shape(points,shape):
    """  returns all points in x that are inside shape """
    assert points.ndim==2 and points.shape[1]==2
    idx = np.all(np.logical_and(points >= 0, points < np.array(shape)), axis=1)
    return points[idx]

def points_to_prob(points, shape, sigma = 1.5,  mode = "max"):
    """points are in (x,y) order!"""

    x = np.zeros(shape, np.float32)
    points = np.asarray(points).astype(np.int32)
    assert points.ndim==2 and points.shape[1]==2

    points = _filter_shape(points, shape[::-1])

    if len(points)==0:
        return x 
    
    if mode == "max":
        D = cdist(points, points)
        A = D < 8*sigma+1
        np.fill_diagonal(A,False) 
        G = nx.from_numpy_array(A)
        x = np.zeros(shape, np.float32)
        while len(G)>0:
            inds = nx.maximal_independent_set(G)
            gauss = np.zeros(shape, np.float32)
            gauss[tuple(points[inds].T[[1,0]])] = 1
            g = gaussian_filter(gauss, sigma, mode=  "constant")
            g /= np.max(g)
            x = np.maximum(x,g)
            G.remove_nodes_from(inds)
                
    elif mode == 'sum':
        x = np.zeros(shape, np.float32)
        for px, py in points:
            x[py, px] = 1
        x = sigma**2*2.*np.pi*gaussian_filter(x, sigma, mode=  "constant")
        
    elif mode =="dist":
        from edt import edt
        from scipy.ndimage import zoom
        from skimage.morphology import binary_dilation, disk
        y = np.zeros(shape, np.bool)
        for px, py in points:
            y[py, px] = True
        y = binary_dilation(y, disk(11))
        x = np.exp(-.4*edt(~y))
        x = zoom(x,(1/4,1/4), order=1)
    else:
        raise ValueError(mode)
        
        
            
    return x


    # Y,X = np.meshgrid(*tuple(np.arange(s) for s in shape), indexing = "ij")

    # cy, cx = shape[0]//2, shape[1]//2    
    # gauss = np.exp(-((X-cx)**2+(Y-cy)**2)/sigma**2/2)
    
    # for px, py in points:
    #     x = np.maximum(x,np.roll(np.roll(gauss, py-cy, axis = 0), px-cx, axis = 1))

    # return x


def prob_to_points(prob, prob_thresh=.5, min_distance = 2, subpix=False):
    assert prob.ndim==2, "Wrong dimension of prob"

    corners = corner_peaks(prob, min_distance = min_distance, threshold_abs = prob_thresh, threshold_rel=0)
    if subpix:
        print("using subpix")
        corners_sub = corner_subpix(prob, corners, window_size=3)
        ind = ~np.isnan(corners_sub[:,0])
        corners[ind] = corners_sub[ind].round().astype(int)
        
    return corners


def points_to_label(points, shape = None, max_distance=3):
    points = np.asarray(points).astype(np.int32)
    assert points.ndim==2 and points.shape[1]==2

    if shape is None:
        mi = np.min(points,axis=0)
        ma = np.max(points,axis=0)
        points = points-mi
        shape = (ma-mi).astype(int)

    im = np.zeros(shape, np.uint16)
    
    for i,p in enumerate(points):
        rr,cc = disk(*p,max_distance, shape = shape)
        im[rr,cc] = i+1
    return im

        
# def points_matching(p1, p2, max_distance=3, report_matches=False):
#     p1 = np.asarray(p1).astype(np.int32)
#     p2 = np.asarray(p2).astype(np.int32)
#     assert p1.ndim==2 and p1.shape[1]==2
#     assert p2.ndim==2 and p2.shape[1]==2
#     if len(p1)==0 or len(p2)==0:
#         return namedtuple("Matching",["accuracy"])(0)        
#         # raise ValueError("empty point set!")
    
#     mi = np.minimum(np.min(p1,axis=0),np.min(p2,axis=0))
#     ma = np.maximum(np.max(p1,axis=0),np.max(p2,axis=0))
#     p1, p2 = p1-mi, p2-mi
#     shape = (ma-mi).astype(int)
#     im1 = points_to_label(p1, shape, max_distance=max_distance)
#     im2 = points_to_label(p2, shape, max_distance=max_distance)
       
#     return matching(im1, im2, thresh = .001, report_matches=report_matches)


def points_matching(p1, p2, cutoff_distance = 5):
    """ finds matching that minimizes sum of mean squared distances"""
    
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    D = cdist(p1,p2, metric='sqeuclidean')

    if D.size>0:
        D[D>cutoff_distance**2] = 1e10*(1+D.max())
    
    i,j = linear_sum_assignment(D)
    valid = D[i,j] <= cutoff_distance**2
    i,j = i[valid], j[valid]
    
    res = SimpleNamespace()
    
    tp = len(i)
    fp = len(p2)-tp
    fn = len(p1)-tp
    res.tp = tp
    res.fp = fp
    res.fn = fn
    res.accuracy  = tp/(tp+fp+fn) if tp > 0 else 0
    res.precision = tp/(tp+fp) if tp > 0 else 0
    res.recall    = tp/(tp+fn) if tp > 0 else 0
    res.f1        = (2*tp)/(2*tp+fp+fn) if tp > 0 else 0
    
    res.dist = np.sqrt(D[i,j])
    res.mean_dist = np.mean(res.dist) if len(res.dist)>0 else 0

    res.false_negatives = tuple(set(range(len(p1))).difference(set(i)))
    res.false_positives = tuple(set(range(len(p2))).difference(set(j)))
    res.matched_pairs = tuple(zip(i,j)) 
    return res

    
def points_matching_dataset(p1s, p2s, cutoff_distance=5, by_image=True):
    """ 
    by_image is True -> metrics are computed by image and then averaged
    by_image is True -> TP/FP/FN are aggregated and only then are metrics computed
    """
    stats = tuple(points_matching(p1,p2,cutoff_distance=cutoff_distance) for p1,p2 in zip(p1s, p2s))


    if by_image:
        res = dict()
        for k, v in vars(stats[0]).items():
            if np.isscalar(v):
                res[k] = np.mean([vars(s)[k] for s in stats])
        return SimpleNamespace(**res)
    else:
        res = SimpleNamespace()
        res.tp = 0 
        res.fp = 0 
        res.fn = 0 


        for s in stats: 
            for k in ('tp','fp', 'fn'):
                setattr(res,k, getattr(res,k) + getattr(s, k))

        res.accuracy  = res.tp/(res.tp+res.fp+res.fn) if res.tp > 0 else 0
        res.precision = res.tp/(res.tp+res.fp) if res.tp > 0 else 0
        res.recall    = res.tp/(res.tp+res.fn) if res.tp > 0 else 0
        res.f1        = (2*res.tp)/(2*res.tp+res.fp+res.fn) if res.tp > 0 else 0

        return res
        
    

def multiscale_decimate(y, decimate = (4,4), sigma = 1):
    if decimate==(1,1):
        return y
    assert y.ndim==len(decimate)
    from skimage.measure import block_reduce
    from scipy.ndimage import gaussian_filter
    y = block_reduce(y, decimate, np.max)
    y = 2*np.pi*sigma**2*gaussian_filter(y,sigma)
    y = np.clip(y,0,1)
    return y


def voronoize(points, shape):
    """ returns distances to closest points"""
    from scipy.spatial.distance import cdist

    assert points.shape[-1]==2

    if len(points)==0:
        return np.array([]), 10000000*np.ones(shape+(2,))
    
    grid  = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing = "ij"),-1).reshape((-1,2))
    
    dist = cdist(grid, points, metric="sqeuclidean")    
    indices = np.argmin(dist, axis=-1)

    dist_closest = (points[indices]-grid).reshape(shape+(2,)).astype(np.float32)
    lbl = indices.reshape(shape)
    
    return indices, dist_closest

def voronoize_from_prob(prob,prob_thresh =0.9):

    points = prob_to_points(prob,prob_thresh =0.9)

    _, dist_closest = voronoize(points,shape= prob.shape)
    
    return dist_closest


def optimize_threshold(Y, Yhat, model, measure='accuracy', bracket=None, tol=1e-4, maxiter=80, verbose=1):
    values = dict()

    if bracket is None:
        bracket = .1, .9
    print("bracket =", bracket)
    values = dict()

    p_gt = tuple(prob_to_points(y, prob_thresh=0.95, min_distance=0) for y in Y)

    with tqdm(total=maxiter, disable=(verbose!=1)) as progress:

        def fn(thr):
            prob_thresh = np.clip(thr, *bracket)
            value = values.get(prob_thresh)
            if value is None:
                p_pred = tuple(prob_to_points(y, prob_thresh=prob_thresh) for y in Yhat)

                # value = np.mean(tuple(points_matching(p1,p2)._asdict()[measure] for p1,p2 in zip(p_gt, p_pred)))

                value = np.mean(tuple(vars(points_matching(p1,p2))[measure] for p1,p2 in zip(p_gt, p_pred)))

                values[prob_thresh]=value
            if verbose > 1:
                print("{now}   thresh: {prob_thresh:f}   {measure}: {value:f}".format(
                    now = datetime.datetime.now().strftime('%H:%M:%S'),
                    prob_thresh = prob_thresh,
                    measure = measure,
                    value = value,
                ), flush=True)
            else:
                progress.update()
                progress.set_postfix_str("{prob_thresh:.3f} -> {value:.3f}".format(prob_thresh=prob_thresh, value=value))
                progress.refresh()
            return -value

        opt = minimize_scalar(fn, method='golden', bracket=bracket, tol=tol, options={'maxiter': maxiter})

    verbose > 1 and print('\n',opt, flush=True)
    return opt.x, -opt.fun


def warp(img, points, amount = 8, grid = 8, order=3):
    """points are a list of (x,y) coordinates!"""
    assert img.ndim==2
    if np.isscalar(grid): grid = (grid,)*2
    
    Y,X = np.meshgrid(*tuple(np.arange(s) for s in img.shape[:2]),indexing = "ij")
    r = np.random.uniform(-1,1,grid)
    # r = amount*(r-np.min(r))/(np.max(r)-np.min(r)+1e-10)
    r = zoom(r,tuple(s2/s1 for s1,s2 in zip(r.shape,img.shape)), order=3)
    # create solenoidal flow field 
    dv = np.stack(np.gradient(r))
    dv = amount*dv/dv.max()
    DY, DX = -dv[1], dv[0]
    V = np.stack([(Y+DY).ravel(),(X+DX).ravel()])
    V_inv = np.stack([(Y-DY).ravel(),(X-DX).ravel()])
    img_warped = map_coordinates(img, V, order=order).reshape(img.shape)
    Y2,X2 = map_coordinates(Y, V_inv, order=order).reshape(Y.shape),map_coordinates(X, V_inv, order=order).reshape(X.shape)
    points_warped = np.stack([X2[tuple(points.T[[1,0]])], Y2[tuple(points.T[[1,0]])]], axis =-1)
    return img_warped, points_warped


def mixseg_warp(img, points, n_copys = 2, amount = 8, grid = 8, order=3):
    xs, ps = zip(*tuple(warp(img, points,amount=amount,grid=grid, order=order) for _ in range(n_copys)))
    xs = np.stack(xs, axis=0)
    xs = xs*np.random.uniform(.5,1.4,(n_copys,1,1))
    x = np.max(xs,axis = 0 )
    p = np.concatenate(ps)
    return x, p


if __name__ == '__main__':

    np.random.seed(42)
    shape = (512,512)
    # points = np.random.randint(0,512,(1000,2))

    # u1 = points_to_prob(points, shape, sigma = 2, use_graph=True)
    # u2 = points_to_prob(points, shape, sigma = 2, use_graph=False)


    points = np.random.randint(0,512,(100,2))
    img = points_to_prob(points, shape, sigma = 2, use_graph=True)

    # img = np.zeros((512,512))
    # img[40:70,40:70] = 1
    

    # img_warped, points_warped = warp(img, points, amount=12, grid=4)

    x, p = mixseg_warp(img, points, 3)

    
    
    
def center_pad(x, shape, mode = "reflect"):
    """ pads x to shape , inverse of center_crop"""
    if x.shape == shape:
        return x
    if not all([s1<=s2 for s1,s2 in zip(x.shape,shape)]):
        raise ValueError(f"shape of x {x.shape} is larger than final shape {shape}")
    diff = np.array(shape)- np.array(x.shape)
    return np.pad(x,tuple((int(np.ceil(d/2)),d-int(np.ceil(d/2))) if d>0 else (0,0) for d in diff), mode=mode)


def center_crop(x, shape):
    """ crops x to shape, inverse of center_pad 

    y = center_pad(x,shape)
    z = center_crop(y,x.shape)
    np.allclose(x,z)
    """
    if x.shape == shape:
        return x
    if not all([s1>=s2 for s1,s2 in zip(x.shape,shape)]):
        raise ValueError(f"shape of x {x.shape} is smaller than final shape {shape}")
    diff = np.array(x.shape)- np.array(shape)
    ss = tuple(slice(int(np.ceil(d/2)),s-d+int(np.ceil(d/2))) if d>0 else slice(None) for d,s in zip(diff,x.shape))
    return x[ss]


def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def str2scalar(dtype):
    def _f(v):
        if v.lower() == "none":
            return None
        else:
            return dtype(v)

    return _f



def normalize_fast2d(x, pmin=1, pmax=99.8, dst_range=(0,1.), clip = False, sub = 4, blocksize=None, order=1, ignore_val=None):
    """
    normalizes a 2d image
    if blocksize is not None (e.g. 512), computes adaptive/blockwise percentiles
    """
    assert x.ndim==2

    out_slice = slice(None),slice(None)
    
    if blocksize is None:
        x_sub = x[::sub,::sub]
        if ignore_val is not None:
            x_sub = x_sub[x_sub!=ignore_val]
        mi, ma = np.percentile(x_sub,(pmin,pmax))#.astype(x.dtype)
        print(f"normalizing_fast with mi = {mi:.2f}, ma = {ma:.2f}")        
    else:
        from csbdeep.internals.predict import tile_iterator_1d
        try:
            import cv2
        except ImportError:
            raise ImportError("normalize_adaptive() needs opencv, which is missing. Please install it via 'pip install opencv-python'")

        if np.isscalar(blocksize):
            blocksize = (blocksize, )*2
            
        if not all(s%b==0 for s,b in zip(x.shape, blocksize)):
            warnings.warn(f"image size {x.shape} not divisible by blocksize {blocksize}")
            pads = tuple(b-s%b for b, s in zip(blocksize, x.shape))
            out_slice = tuple(slice(0,s) for s in x.shape)
            print(f'padding with {pads}')
            x = np.pad(x,tuple((0,p) for p in pads), mode='reflect')

        print(x.shape)
        n_tiles = tuple(max(1,s//b) for s,b in zip(x.shape, blocksize))
        
        print(f"normalizing_fast adaptively with {n_tiles} tiles and order {order}")        
        mi, ma = np.zeros(n_tiles, x.dtype), np.zeros(n_tiles, x.dtype)

        kwargs=dict(block_size=1, n_block_overlap=0, guarantee="n_tiles")

        for i, (itile,is_src,is_dst) in enumerate(tile_iterator_1d(x, axis=0,
                                                                   n_tiles=n_tiles[0], **kwargs)):
            for j, (tile,s_src,s_dst) in enumerate(tile_iterator_1d(itile, axis=1,
                                                                    n_tiles=n_tiles[1], **kwargs)):
                x_sub = tile[::sub,::sub]
                if ignore_val is not None:
                    x_sub = x_sub[x_sub!=ignore_val]
                    x_sub = np.array(0) if len(x_sub)==0 else x_sub
                mi[i,j], ma[i,j] = np.percentile(x_sub,(pmin,pmax)).astype(x.dtype)

        interpolations = {0:cv2.INTER_NEAREST,
                          1:cv2.INTER_LINEAR}

        mi = cv2.resize(mi, x.shape[::-1], interpolation=interpolations[order])
        ma = cv2.resize(ma, x.shape[::-1], interpolation=interpolations[order])

    x = x.astype(np.float32)
    x -= mi
    x *= (dst_range[1]-dst_range[0])
    x /= ma-mi+1e-20
    x = x[out_slice]
    
    x += dst_range[0]
    
    if clip:
        x = np.clip(x,*dst_range)
    return x
