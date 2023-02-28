from types import SimpleNamespace
import numpy as np
import sys
from tqdm import tqdm 
from collections import namedtuple
from itertools import product
import tensorflow as tf
from scipy.ndimage import zoom
import tensorflow as tf

from csbdeep.internals.train import ParameterDecayCallback
from csbdeep.internals.predict import tile_iterator
from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict, load_json, save_json
from csbdeep.utils.tf import CARETensorBoardImage
from csbdeep.models import CARE
from csbdeep.models import Config as CareConfig
from csbdeep.internals.train import RollingSequence

from .multiscalenet import multiscale_unet
from .hrnet import hrnet
from .utils import _filter_shape, points_to_flow, prob_to_points, points_to_prob, points_matching, optimize_threshold, points_matching, multiscale_decimate, voronoize_from_prob, center_pad, center_crop

def weighted_bce_loss(extra_weight=1, thr=0.05):
    def _loss(y_true,y_pred):
        # mask_gt   = tf.keras.backend.cast(y_true[...,:1]>=thr, tf.keras.backend.floatx())
        # mask_pred = tf.keras.backend.cast(y_pred[...,:1]>=thr, tf.keras.backend.floatx())
        # mask = tf.math.maximum(mask_gt , mask_pred)
        mask = tf.keras.backend.cast(y_true[...,:1]>=thr, tf.keras.backend.floatx())
        loss = (1+extra_weight*mask)*tf.keras.backend.binary_crossentropy(y_true, y_pred)
        return loss
    return _loss

def focal_loss(extra_weight=1, gamma=2):
    def _loss(y_true,y_pred):
        mask = tf.keras.backend.cast(y_true>0.001, tf.keras.backend.floatx())

        eps  = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, eps, 1.0 - eps)
        # Calculate cross entropy
        ce1 = -tf.math.log(y_pred)
        w1 = tf.math.pow((1 - y_pred), gamma)
        ce2 = -tf.math.log(1-y_pred)
        w2 = tf.math.pow(y_pred, gamma)
        loss = y_true*ce1*w1 + (1-y_true)*ce2*w2

        loss = (1+extra_weight*mask)*loss
        return loss        
    return _loss



def weighted_mae_loss(extra_weight=1, thr=0.05):
    def _loss(y_true,y_pred):
        mask = tf.keras.backend.cast(y_true[...,:1]>thr, tf.keras.backend.floatx())
        loss = (1+extra_weight*mask)*tf.keras.backend.abs(y_true-y_pred)
        return loss        
    return _loss

def weighted_mse_loss(extra_weight=1, thr=0.05):
    def _loss(y_true,y_pred):
        mask   = tf.keras.backend.cast(y_true[...,:1]>=thr, tf.keras.backend.floatx())
        loss = (1+extra_weight*mask)*tf.keras.backend.square(y_true-y_pred)
        return loss        
    return _loss


def scale_sum(extra_weight=1, sigmas = (0, 2, 8), powers = (1,2,4) ):
    # extra weight mask with smoothing (that will weight clusters according to the local density of spots)
    mask_sigma0 = 1
    mask_sigma  = 6
    # smoothing parameters as y_true is already smoothed
    mask_alpha, mask_sigma = (mask_sigma/mask_sigma0)**2, np.sqrt(mask_sigma**2-mask_sigma0**2)
    
    def _gauss(sigma,axis = 0 , norm =True):
        n = int(2*sigma)+1
        if sigma>1e-2:
            h = np.exp(-np.arange(-n,n+1)**2/2/sigma**2).astype(np.float32)
        else:
            h = np.ones(1,np.float32)
            
        if norm is False or norm is None:
            pass
        elif norm is True:
            h /= np.sum(h)
        elif norm=="sqrt":
            h /= np.sqrt(np.sum(h))
        else:
            ValueError(norm)
            
        if axis ==0:
            shape = (len(h),1,1,1)
        elif axis ==1:
            shape = (1,len(h),1,1)
        else:
            raise ValueError(axis)

        return tf.constant(h.astype(np.float32).reshape(shape))

    def _gauss_pair(sigma, norm):
        return _gauss(sigma, axis = 0, norm=norm), _gauss(sigma, axis = 1, norm=norm)
    
    # hs = tuple(_gauss_pair(s, norm="sqrt") for s in sigmas)
    # hs = tuple(_gauss_pair(s, norm="sqrt") for s in sigmas)
    hs = tuple(_gauss_pair(s, norm="sqrt") for s in sigmas)
    hs_mask = _gauss_pair(mask_sigma, norm=False) 
    
    def _conv(y, hs, power=1):
        # if power != 1:
        #     y = tf.keras.backend.minimum(y, 1)
        tmp = tf.pow(y,power)
        tmp = tf.nn.conv2d(tmp,hs[0], strides=(1,1), padding="SAME")
        tmp = tf.nn.conv2d(tmp,hs[1], strides=(1,1), padding="SAME")
        return tmp

    def _loss(y_true,y_pred):
        mask = tf.keras.backend.clip(_conv(y_true, hs_mask,power=1),0,1)

        y_true = tf.keras.backend.minimum(y_true, 1)
        y_pred = tf.keras.backend.minimum(y_pred, 1)
        
        y_true_sums = tuple(_conv(y_true, h, power) for h,power in product(hs,powers))
        y_pred_sums = tuple(_conv(y_pred, h, power) for h,power in product(hs,powers))
    
        loss = sum(tf.keras.backend.abs(a-b) for a,b in zip(y_true_sums,y_pred_sums))
        # loss /= len(y_true_sums)
        loss = loss * (1+extra_weight*mask)
        return loss
    
    return _loss


def masked_distance_loss(cutoff_distance):
    def _loss(y_true,y_pred):
        squared_dist = tf.keras.backend.sum(tf.keras.backend.square(y_true), axis = -1, keepdims = True)
        mask = tf.keras.backend.cast(squared_dist<=cutoff_distance**2, tf.keras.backend.floatx())
        loss = mask*tf.keras.backend.abs(y_true - y_pred)
        loss = loss / (tf.keras.backend.epsilon()+tf.keras.backend.mean(mask))        
        return loss        
    return _loss


def dice_loss(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    num   = 2 * tf.reduce_sum(y_true * y_pred)
    denom = tf.reduce_sum(y_true + y_pred) + 1e-10
    loss = 1 - num/denom
    return loss


class AccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, spot_model, tb_callback, X, P):
        super().__init__()
        self.X, self.P = X, P
        self.spot_model = spot_model
        self.tb_callback = tb_callback
        self.accs = []
        self.f1s = []

    def on_epoch_end(self, epoch, logs={}):
        p_pred, p_details  = tuple(zip(*tuple(self.spot_model.predict(x, verbose=False, return_details=True) for x in self.X)))
        
        stats = tuple(points_matching(p1,p2, cutoff_distance=3) for p1,p2 in zip(self.P, p_pred))
        
        ac = np.mean(tuple(s.accuracy for s in stats))  
        f1 = np.mean(tuple(s.f1 for s in stats))

        self.accs.append(ac)
        self.f1s.append(f1)

        spot_prob = np.concatenate(tuple(details.heatmap[tuple(p1.T.astype(int))] for p1, details in zip(self.P, p_details)), axis=0)
        spot_prob = spot_prob.mean()

        print(p_pred[0].shape)
        print(f"VALIDATION: val_acc : {ac:.3f}  val_f1 : {f1:.3f}  val_spot_prob : {spot_prob:.3f} ")

            
        if self.tb_callback is not None:
            with self.tb_callback._writers["val"].as_default():
                tf.summary.scalar('val_F1', f1, step=epoch)
                tf.summary.scalar('val_spot_prob', spot_prob, step=epoch)


class Config(CareConfig):
    def __init__(self,axes = "YX", loss = "bce", n_channel_in = 1, remove_bkg = False, spot_sigma=1.5, spot_mode='max', unet_n_depth = 3, train_loss_weights = (1,1,1), spot_weight = 5,  spot_weight_decay=.1 , backbone="unet", activation="relu", train_batch_norm=False, fuse_heads=False, foreground_prob=.3, patch_size=(256,256), multiscale_loss_decay_exponent=2, **kwargs):
        kwargs.setdefault("train_batch_size",2)
        kwargs.setdefault("train_reduce_lr", {'factor': 0.5, 'patience': 40})
        kwargs.setdefault("n_channel_in",n_channel_in)
        kwargs.setdefault("n_channel_out",1)
        kwargs.setdefault("unet_kern_size",3)
        kwargs.setdefault("unet_n_filter_base",32)
        kwargs.setdefault("unet_pool",2)
        super().__init__(axes=axes, unet_n_depth=unet_n_depth,
                         allow_new_parameters=True, **kwargs)
        self.spot_weight = spot_weight
        self.spot_sigma = spot_sigma
        self.spot_mode = spot_mode
        self.spot_weight_decay = spot_weight_decay
        self.multiscale_loss_decay_exponent = multiscale_loss_decay_exponent

        self.patch_size = patch_size
        self.foreground_prob = foreground_prob
        self.train_batch_norm      = train_batch_norm
        self.fuse_heads = fuse_heads
        self.activation = activation
        self.remove_bkg = remove_bkg
        self.train_loss_weights = train_loss_weights

        assert backbone in ('unet', )
        self.backbone = backbone

        if spot_mode in ('max', 'sum'):
            self.spot_mode = spot_mode
        else:
            raise ValueError('spot  mode must be max or sum')

        if loss in ("mae", "mse", "bce"):
            self.loss = loss
        else:
            raise ValueError(loss)


class SpotNetData(RollingSequence):
    """ creates a generator from data"""
    def __init__(self, X,P, patch_size, length, batch_size=4, spot_mode='max', sigma=2, shuffle=True, augmenter = None):
        """datas has to be a tuple of all input/output lists, e.g.  
        datas = (X,Y)
        """

        nD = len(patch_size)
        assert nD==2
        x_ndim = X[0].ndim
        assert x_ndim in (nD,nD+1)

        assert len(X)==len(P)

        if not spot_mode in ('max', 'sum'):
            raise NotImplementedError(f'unknown mode {spot_mode}')
        
        if x_ndim == nD:
            self.n_channel = None
        else:
            self.n_channel = X[0].shape[-1]

        super().__init__(data_size=len(X),
                         batch_size=batch_size, length=length, shuffle=shuffle)

        self.spot_mode = spot_mode
        self.X, self.P = X, P
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.sigma = sigma
        self.augmenter = augmenter

    def _sample_patch(self, k):
        x, p = self.X[k], self.P[k]
        if any(p>s for p,s in zip(self.patch_size, x.shape)):
            raise ValueError("patch size cannot be larger than image size")
        offset = np.array([np.random.randint(0, s-p+1) for s,p in zip(x.shape[:2], self.patch_size)])
        slices = tuple(slice(s,s+p) for s,p in zip(offset, self.patch_size))
        p = p-offset
        points = _filter_shape(p, self.patch_size)
        return x[slices], points


    def __getitem__(self, i):
        idx = self.batch(i)

        X, P = tuple(zip(*tuple(self._sample_patch(k) for k in idx)))
        
        if self.augmenter is not None:
            X, P = tuple(zip(*tuple(self.augmenter(x, p) for x, p in zip(X, P))))

        F = tuple(points_to_flow(p, x.shape[:2], sigma=self.sigma) for x, p in zip(X,P))

        U = tuple(np.expand_dims(points_to_prob(p, x.shape[:2], mode=self.spot_mode, sigma=self.sigma), -1) for x, p in zip(X,P))

        X = np.stack(X)
        U = np.stack(U)
        F = np.stack(F)

        if X.ndim == 3: # input image has no channel axis
            X = np.expand_dims(X,-1)
        return X, (U, F)


class RemoveBackground(tf.keras.layers.Layer):
    def __init__(self, n_channels=1, radius=51):
        super().__init__()

        assert n_channels==1
        assert radius%2==1

        self.half_radius = int(radius)//2

        h = np.exp(-np.linspace(-2,2,2*self.half_radius+1)**2).astype(np.float32)
        h /= np.sum(h)
        wy = h.reshape((1, len(h),1,1))
        wx = h.reshape((len(h),1, 1,1))
        self.wy = tf.Variable(initial_value=tf.convert_to_tensor(wy), trainable=False, name='remove_bkg_y')
        self.wx = tf.Variable(initial_value=tf.convert_to_tensor(wx), trainable=False, name='remove_bkg_x')

    def call(self, x):
        y = tf.pad(x, [[0,0], [self.half_radius,self.half_radius], [self.half_radius,self.half_radius], [0,0]], mode='REFLECT')
        y = tf.nn.conv2d(y, self.wy, strides=[1, 1, 1, 1], padding='VALID')
        y = tf.nn.conv2d(y, self.wx, strides=[1, 1, 1, 1], padding='VALID')
        x = x - y
        return x

class SpotNet(CARE):

    def __init__(self, config, name=None, basedir='.'):
        super().__init__(config=config, name=name, basedir=basedir)
        threshs = dict(prob=None)
        if basedir is not None:
            try:
                threshs = load_json(str(self.logdir / 'thresholds.json'))
                print("Loading thresholds from 'thresholds.json'.")
                if threshs.get('prob') is None or not (0 < threshs.get('prob') < 1):
                    print("- Invalid 'prob' threshold (%s), using default value." % str(threshs.get('prob')))
                    threshs['prob'] = None
            except FileNotFoundError:
                if config is None and len(tuple(self.logdir.glob('*.h5'))) > 0:
                    print("Couldn't load thresholds from 'thresholds.json', using default values. "
                          "(Call 'optimize_thresholds' to change that.)")

        default_thresh = 0 if self.config.spot_mode == 'flow' else 0.5

        self.thresholds = dict (
            prob = default_thresh if threshs['prob'] is None else threshs['prob'],
        )
        print("Loaded/default threshold values: prob_thresh={prob:g}".format(prob=self.thresholds.prob))

    @property
    def thresholds(self):
        return self._thresholds


    @thresholds.setter
    def thresholds(self, d):
        self._thresholds = namedtuple('Thresholds',d.keys())(*d.values())
        
    def _build(self):
        
        if self.config.backbone=='unet':
            model, scales = multiscale_unet(
                input_shape = (None,None,self.config.n_channel_in),
                n_depth=self.config.unet_n_depth,
                n_filter_base=self.config.unet_n_filter_base,
                kernel_size = self.config.unet_kern_size,
                fuse_heads=self.config.fuse_heads,
                pool_size=self.config.unet_pool,
                activation=self.config.activation,
                last_activation='sigmoid'
            )
            scales = (1,) + tuple(scales)

        else: 
            raise ValueError(f"Unknown backbone: {self.config.backbone}")
        
        self.multiscale_factors = scales

        inp = tf.keras.layers.Input((None,None, self.config.n_channel_in))

        if self.config.remove_bkg:    
            x = RemoveBackground(radius=51)(inp)
            out = model(x)
        else: 
            out = model(inp)

        model = tf.keras.models.Model(inp, out)

        return model

    def _prepare_target_multiscale(self, y):
        """returns a tuple of multiscales (including the first scale)"""
        
        # image without channel (h, w)
        if y.ndim==2:
            return tuple(multiscale_decimate(y,(fac,fac)) for fac in self.multiscale_factors)
        # image with channel (h, w, 1)
        elif y.ndim==3 and y.shape[-1]==1:
            return tuple(multiscale_decimate(y[...,0],(fac,fac))[...,np.newaxis] for fac in self.multiscale_factors)
        # batch without channel (b, h , w)
        elif y.ndim==3 and y.shape[-1]>1:
            return tuple(np.stack([multiscale_decimate(_y,(fac,fac))  for _y in y]) for fac in self.multiscale_factors)
        # batch  with channel (b, h, w, 1)
        elif y.ndim==4 and y.shape[-1]==1:
            return tuple(np.stack([multiscale_decimate(_y[...,0],(fac,fac))[...,np.newaxis]  for _y in y]) for fac in self.multiscale_factors)
        else:
            raise ValueError(f"unsupported dimension {y.ndim} (shape {y.shape})")


    def prepare_targets(self, y):
        """ from a single output (batch), generate a list of additional targets"""
        return list(self._prepare_target_multiscale(y))

    
    @property
    def _config_class(self):
        return Config

    def prepare_for_training(self, optimizer=None):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.train_learning_rate)
        # self.keras_model.compile(optimizer, loss="binary_crossentropy")

        lw_final, lw_multiscale, lw_flow = self.config.train_loss_weights

        

        # Loss for final layer 
        weight = tf.keras.backend.variable(self.config.spot_weight)
        if self.config.loss == "bce":
            loss = [weighted_bce_loss(weight)]
        elif self.config.loss == "mse":
            loss = [weighted_mse_loss(weight, thr=-.8)]
        elif self.config.loss == "mae":
            loss = [weighted_mae_loss(weight, thr=-.8)]
            raise ValueError(self.config.mode)
        loss_weights = [lw_final]
        
        # Loss for multiscale layers 
        loss += [tf.keras.backend.binary_crossentropy]*(len(self.multiscale_factors)-1)
        loss_weights += list(lw_multiscale/np.array(self.multiscale_factors[1:])**self.config.multiscale_loss_decay_exponent)

        # loss for flow 
        loss += [tf.keras.losses.mean_absolute_error]
        loss_weights += [lw_flow]

        print("loss_weights ", loss_weights)
        self.keras_model.compile(optimizer, loss=loss, loss_weights = loss_weights)
 
        self.callbacks = []

        self.callbacks.append(ParameterDecayCallback(weight, self.config.spot_weight_decay, name='extra_spot_weight', verbose=True))

        self.tb_callback = None
        if self.basedir is not None:
            self.callbacks += self._checkpoint_callbacks()

            if self.config.train_tensorboard:
                from tensorflow.keras.callbacks import TensorBoard
                self.tb_callback = TensorBoard(log_dir=str(self.logdir/'logs'), write_graph=False, profile_batch=0)
                self.callbacks.append(self.tb_callback)


                
        if self.config.train_reduce_lr is not None:
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            self.callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True

    def train(self, X, P, validation_data, augmenter = None, epochs=None, steps_per_epoch=None, workers=1):
        """Train the neural network with the given data.
        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of source images.
        P : :class:`numpy.ndarray`
            Array of target coordinates.
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of arrays for source images and target coordinates
        augmenter : function that takes an image pair (x,y) as input 
            Tuple of arrays for source and target validation images.
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.
        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.
        """


        ((isinstance(validation_data,(list,tuple)) and len(validation_data)==2)
            or _raise(ValueError('validation_data must be a pair of numpy arrays')))
        
        assert len(X)==len(P)
        assert X[0].ndim == (2 if self.config.n_channel_in==1 else 3)

        Xv, Pv = validation_data 
        
        if self.config.n_channel_in==1:
            axes = self.config.axes.replace('C','')
        else:
            axes = self.config.axes
        axes = axes_check_and_normalize('S'+axes,X[0].ndim+1)
        ax = axes_dict(axes)

        for a,div_by in zip(axes,self._axes_div_by(axes)):
            n = X[0].shape[ax[a]-1]
            if n % div_by != 0:
                raise ValueError(
                    "training images must be evenly divisible by %d along axis %s"
                    " (which has incompatible size %d)" % (div_by,a,n)
                )

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()
            
        self.data = SpotNetData(X, P,
                                augmenter = augmenter,
                                shuffle=True,
                                spot_mode=self.config.spot_mode,
                                length = epochs*steps_per_epoch,
                                sigma=self.config.spot_sigma,
                                patch_size=self.config.patch_size,
                                batch_size=self.config.train_batch_size)
        
        self.data_val = SpotNetData(Xv, Pv,
                                batch_size=max(16,len(Xv)), length=1,
                                shuffle=False,
                                spot_mode=self.config.spot_mode,
                                patch_size=self.config.patch_size,
                                sigma=self.config.spot_sigma,
                                )
        
        
        
        def _transform(u,f):
            return list(self.prepare_targets(u)) + [f]                               

        def _copy_gen(gen):
            for x, (y, f) in gen:
                yield x , _transform(y,f)

        validation_data = self.data_val[0][0], _transform(*self.data_val[0][1])
        self.data = _copy_gen(self.data)
        self.data_val = validation_data

        data_train = iter(self.data)

        callbacks = self.callbacks
        self.callbacks.append(AccuracyCallback(self, self.tb_callback, Xv, Pv))

        if (self.config.train_tensorboard and self.basedir is not None and
            not any(isinstance(cb,CARETensorBoardImage) for cb in callbacks)):
            callbacks.append(CARETensorBoardImage(model=self.keras_model, data=validation_data,
                                                  log_dir=str(self.logdir/'logs'/'images'),
                                                  n_images=3))
            

        history = self.keras_model.fit(data_train,
                                       validation_data=tuple(validation_data),
                                       epochs=epochs,
                                       steps_per_epoch=steps_per_epoch,
                                       callbacks=callbacks,
                                       validation_batch_size=self.config.train_batch_size,
                                       verbose=1)

        self._training_finished()

        return history


    def _axes_div_by(self, query_axes):
        query_axes = axes_check_and_normalize(query_axes)
        pool_div_by = self.config.unet_pool**self.config.unet_n_depth
        return tuple((pool_div_by if a in 'XYZT' else 1) for a in query_axes)

    def _predict_prob(self, x, **kwargs):
        assert x.ndim ==3 and x.shape[-1]==self.config.n_channel_in
        u = self.keras_model.predict(x[np.newaxis], **kwargs)
        u = u[0] 
        return u[0,...,0]

    
    def predict(self, img, prob_thresh=None, n_tiles=(1,1), subpix = False, min_distance=2, scale = None, 
                    normalizer=None, return_details = False, peak_mode='skimage',
                    verbose=True, show_tile_progress=False):

        if img.ndim==2:
            img = img[...,np.newaxis]
        if not img.shape[-1]==self.config.n_channel_in:
            raise ValueError(f"img should be of shape (m,n,{self.config.n_channel_in}) but is {img.shape}!")

        if scale is not None:
            if verbose: print(f"zooming with scale {scale}")
            x = zoom(img, (scale,scale,1), order=1)
        else:
            x = img
        
        if prob_thresh is None: prob_thresh = self.thresholds.prob
        
        if verbose: print(f"Predicting with prob_thresh = {prob_thresh} and min_distance = {min_distance}")

        div_by = self._axes_div_by("YXC")
        pad_shape = tuple(int(d*np.ceil(s/d)) for s,d in zip(x.shape, div_by))
        if verbose: print(f"padding to shape {pad_shape}")
        x = center_pad(x, pad_shape, mode="reflect")

        if all(n<=1 for n in n_tiles):
            if callable(normalizer):
                x = normalizer(x)

            y = self._predict_prob(x, verbose=verbose)
            if scale is not None:
                print('zooming')
                y = zoom(y, (1./scale,1./scale), order=1)
                    
            y = center_crop(y, img.shape[:2])
            if verbose: print(f"peak detection with prob_thresh={prob_thresh:.2f}, subpix={subpix}, min_distance={min_distance} ...")
            points = prob_to_points(y, prob_thresh=prob_thresh, subpix=subpix, min_distance=min_distance, mode=peak_mode)
                        
        else:
            # output array
            if return_details:
                y = np.empty(x.shape[:2], np.float32)

            points = []
            iter_tiles = tile_iterator(x, n_tiles  = n_tiles +(1,),
                                 block_sizes = div_by,
                                 n_block_overlaps= (2,2,0))
            if callable(show_tile_progress):
                iter_tiles = show_tile_progress(iter_tiles)
            else:
                iter_tiles = tqdm(iter_tiles, total = np.prod(n_tiles),
                                  disable=not show_tile_progress)

            for tile, s_src, s_dst in iter_tiles:
                if callable(normalizer):
                    tile = normalizer(tile)
                y_tile = self._predict_prob(tile, verbose=False)
                y_tile_sub = y_tile[s_src[:2]]

                if return_details:
                    y[s_dst[:2]] = y_tile_sub

                p = prob_to_points(y_tile_sub, prob_thresh=prob_thresh, subpix=subpix, min_distance=min_distance)
                p += np.array([s.start for s in s_dst[:2]])[None]
                points.append(p)

            if return_details:
                if scale is not None:
                    if verbose: print('zooming')
                    y = zoom(y, (1./scale,1./scale), order=1)
                y = center_crop(y, img.shape[:2])

            points = np.concatenate(points, axis=0)
            
        points = _filter_shape(points, img.shape[:2])

        if verbose: print(f"detected {len(points)} points")

        if return_details:
            probs = y[tuple(points.T)]
            details = SimpleNamespace(prob = probs, heatmap = y)
        else: 
            details = SimpleNamespace()

        return points, details
            
        
    def optimize_thresholds(self, X_val, P_val,  verbose=1, predict_kwargs=None, optimize_kwargs=None, save_to_json=True):
        """Optimize two thresholds (probability) necessary for predicting object instances.
        Parameters
        ----------
        X_val : list of ndarray
            (Validation) input images (must be normalized) to use for threshold tuning.
        P_val : list of ndarray
            (Validation) coordinates to use for threshold tuning.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of this class.
            (If not provided, will guess value for `n_tiles` to prevent out of memory errors.)
        optimize_kwargs: dict
            Keyword arguments for ``utils.optimize_threshold`` function.

        """
        if predict_kwargs is None:
            predict_kwargs = {}
        if optimize_kwargs is None:
            optimize_kwargs = {}

        Y_val = [points_to_prob(p, x.shape[:2], sigma=self.config.train_spot_sigma) for x, p in zip(X_val, P_val)]

        def _pad_dims(x):
            return (x if x.ndim==3 else np.expand_dims(x,-1))
        X_val = map(_pad_dims,X_val)
        
        Yhat_val = [self._predict_prob(x) for x in X_val]

        

        opt_prob_thresh, opt_measure = optimize_threshold(P_val, Yhat_val, verbose=verbose, **optimize_kwargs)

       
        opt_threshs = dict(prob=opt_prob_thresh)

        self.thresholds = opt_threshs
        print(end='', file=sys.stderr, flush=True)
        print("Using optimized values: prob_thresh={prob:g}.".format(prob=self.thresholds.prob))
        if save_to_json and self.basedir is not None:
            print("Saving to 'thresholds.json'.")
            save_json(opt_threshs, str(self.logdir / 'thresholds.json'))
        return opt_threshs


    def evaluate(self, img, points_gt, **kwargs):

        u, points = self.predict(img, **kwargs)
        
        return points_matching(points_gt[:,[1,0]], points)
    
