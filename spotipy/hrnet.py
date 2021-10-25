"""


https://github.com/niecongchong/HRNet-keras-semantic-segmentation


"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, add, concatenate


def conv(x, out_filters, kernel=3, strides=(1, 1), batch_norm=False):
    x = Conv2D(out_filters, kernel, padding='same', strides=strides, use_bias=not batch_norm, kernel_initializer='he_normal')(x)
    if batch_norm:  x = BatchNormalization(axis=-1)(x)
    return x


def basic_block(inp, out_filters, strides=(1, 1), num=2, activation='relu', batch_norm=False):

    x = inp
    
    if strides != (1,1) or inp.shape[-1] != out_filters:
        inp = conv(inp, out_filters, 1, strides=strides,
                     batch_norm=batch_norm)

    
    for i in range(num):
        x = conv(x, out_filters, 3, strides=strides if i==0 else (1,1),
                 batch_norm=batch_norm)
        if i<num-1: x = Activation(activation)(x)
    
    x = add([x, inp])

    x = Activation(activation)(x)
    return x


def single_block(input, out_filters, strides=(1, 1), activation='relu', batch_norm=False):
    x = conv(input, out_filters, kernel=3, strides=strides, batch_norm=batch_norm)
    x = Activation(activation)(x)
    return x

def bottleneck_block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x



#---------------

def make_branches(x, filters=(32,64), num=2, batch_norm=False):
    """ parallel branches """
    assert len(x)==len(filters)

    for _ in range(num):
        x = [basic_block(_x, _f, batch_norm=batch_norm) for _x,_f in zip(x,filters)]
    return x


def fuse_layer(x, filters=(32,64), strides=(2,2), batch_norm=False):
    """ mixes all layers in x pairwise, if n=len(filters) < len(x), only return the first n layers"""
    assert len(x)>=len(filters)
    
    def _up(x, nfilters=32, n=1):
        size = tuple(s**n for s in strides)
        x = conv(x, nfilters, 1, batch_norm=batch_norm)
        x = UpSampling2D(size)(x)
        return x
    
    def _down(x, nfilters=32, n=1):
        size = tuple(s**n for s in strides)
        x = conv(x, nfilters, 3, batch_norm=batch_norm)
        x = AveragePooling2D(size)(x)
        return x
    
    def _up_or_down(x, nfilters=32,n=1):
        assert n !=0
        return _up(x,nfilters,-n) if  n<0 else _down(x,nfilters,n)

    y = [] 
    for i,(x1,f) in enumerate(zip(x,filters)):
        y.append(add([x1]+[_up_or_down(x2,f,i-j) for j, x2 in enumerate(x) if i!=j]))

    return y      
        
        
    
def transition_layer(x, filters=(32, 64), batch_norm=False):
    assert len(x)+1==len(filters)
    x = list(x)+[x[-1]]
    x = [basic_block(_x,_f, strides=(1,1) if i<len(x)-1 else (2,2), batch_norm=batch_norm) for i,(_x,_f) in enumerate(zip(x, filters))]

    return x



def hrnet(input_shape=(None,None,1), last_activation='sigmoid',
          n_depth=2, n_filter_base=32, batch_norm=False,
          num_conv_per_depth=2, n_channel_out=1, multi_head=False):
    
    inp = Input(input_shape)

    filters = (n_filter_base,)
    
    x = [inp]
    x = make_branches(x, filters, num=1, batch_norm=batch_norm)

    for i in range(n_depth):
        filters = filters + (int(np.round(np.sqrt(2)*filters[-1])),)
        
        x = transition_layer(x, filters, batch_norm=batch_norm)
        x = make_branches(x, filters, num=1, batch_norm=batch_norm)
        x = fuse_layer(x, filters, batch_norm=batch_norm)
        x = make_branches(x, filters, num=1, batch_norm=batch_norm)

        
    x = make_branches(x, (n_filter_base,)*len(x),  num=1, batch_norm=batch_norm)

    if multi_head:
        out = [single_block(_x, n_channel_out, activation=last_activation) for _x in x]
    else:
        x = fuse_layer(x, filters=(n_filter_base,))[0]
        out = single_block(x, n_channel_out, activation=last_activation)
        
    
    model = Model(inputs=inp, outputs=out)

    return model
    


if __name__ == '__main__':
    # from csbdeep.utils.tf import limit_gpu_memory
    # limit_gpu_memory(1, total_memory=48000, allow_growth=True)

    model = hrnet((64,64,1), n_depth=3, multi_head=True, n_filter_base=32)    
    model.summary()


    
    import numpy as np
    model = hrnet((64,64,1), n_depth=4, multi_head=False, n_filter_base=32)

    x = np.ones((128,64,64,1))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-3))
    model.fit(x,x,epochs=100)
    


    # import numpy as np
    # u = model.predict(np.random.uniform(0,1,(1,128,128,1)))
