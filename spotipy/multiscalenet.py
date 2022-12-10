
import numpy as np
from csbdeep.internals.blocks import conv_block2, resnet_block
from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict, load_json, save_json
import tensorflow as tf


def multiscale_unet(
        input_shape,
        n_channel_out=1,
        n_depth=3,
        n_filter_base=32,
        kernel_size=3,
        fuse_heads=False,
        last_activation="linear",
        n_conv_per_depth=2,
        activation="relu",
        batch_norm=False,
        dropout=0.0,
        pool_size=2,
        kernel_init="he_normal",
        prefix=''):

    conv_block = conv_block2
    pooling    = tf.keras.layers.MaxPooling2D
    upsampling = tf.keras.layers.UpSampling2D

    if last_activation is None:
        last_activation = activation

    channel_axis = -1 if backend_channels_last() else 1

    def _name(s):
        return prefix+s

    skip_layers = []
    multiscale_layers = []
    multiscale_factors = []

    inp = tf.keras.layers.Input(input_shape)

    layer = inp
    # down ...
    for n in range(n_depth):
        for i in range(n_conv_per_depth):
            layer = conv_block(n_filter_base * 2 ** n, kernel_size, kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   init=kernel_init,
                                   batch_norm=batch_norm, name=_name("down_level_%s_no_%s" % (n, i)))(layer)
        skip_layers.append(layer)
        layer = pooling(pool_size, name=_name("max_%s" % n))(layer)

    # middle
    for i in range(n_conv_per_depth - 1):
        layer = conv_block(n_filter_base * 2 ** n_depth, kernel_size, kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=activation,
                               batch_norm=batch_norm, name=_name("middle_%s" % i))(layer)

    layer = conv_block(n_filter_base * 2 ** max(0, n_depth - 1), kernel_size, kernel_size,
                           dropout=dropout,
                           activation=activation,
                           init=kernel_init,
                           batch_norm=batch_norm, name=_name("middle_%s" % n_conv_per_depth))(layer)

    # ...and up with skip layers
    for n in reversed(range(n_depth)):

        # multi head
        multi = layer 
        for i in range(n_conv_per_depth):
            multi = conv_block(n_filter_base * 2 ** n, kernel_size, kernel_size,
                                   dropout=dropout,
                                   init=kernel_init,
                                   activation=activation,
                                   batch_norm=batch_norm, name=_name("multi_level_%s_no_%s" % (n, i)))(multi)
        multi = conv_block(n_channel_out,1,1,activation=last_activation, dtype='float32', name =_name("multi_%s" %n))(multi)

        multiscale_layers.append(multi)
        multiscale_factors.append(pool_size**(n+1))

        # normal unet head 

        layer = tf.keras.layers.Concatenate(axis=channel_axis)([
            upsampling(pool_size)(layer), skip_layers[n], upsampling(pool_size)(layer)
        ])

        for i in range(n_conv_per_depth):
            layer = conv_block(n_filter_base * 2 ** n, kernel_size, kernel_size,
                                   dropout=dropout,
                                   init=kernel_init,
                                   activation=activation,
                                   batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, i)))(layer)

        layer = conv_block(n_filter_base * 2 ** max(0, n - 1), kernel_size, kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=activation,
                               batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, n_conv_per_depth)))(layer)

    if fuse_heads:
        up_multiscale_layers = list(upsampling((s,s), interpolation='bilinear')(u) for s,u in zip(multiscale_factors[::-1], multiscale_layers[::-1]))
        layer = tf.keras.layers.Concatenate()([layer]+up_multiscale_layers)
        
    final = conv_block(n_channel_out, 1,1, activation=last_activation, dtype='float32', name=_name("final"))(layer)
    final_flow = conv_block(3, 1,1, activation='linear', dtype='float32', name=_name("final_flow"))(layer)

    multiscale_factors.append(1)
    
    multiscale_layers = multiscale_layers[::-1]
    multiscale_factors = multiscale_factors[::-1]
    
    return tf.keras.models.Model(inp, [final]+multiscale_layers+[final_flow]), tuple(multiscale_factors)




def multiscale_resunet(
        input_shape,
        n_channel_out=1,
        n_depth=3,
        n_filter_base=64,
        kernel_size=3,
        last_activation="linear",
        n_conv_per_depth=2,
        activation="relu",
        batch_norm=False,
        dropout=0.0,
        pool_size=2,
        kernel_init="he_normal",
        prefix=''):

    conv_block = conv_block2
    pooling    = tf.keras.layers.MaxPooling2D
    upsampling = tf.keras.layers.UpSampling2D

    if last_activation is None:
        last_activation = activation

    channel_axis = -1 if backend_channels_last() else 1

    def _name(s):
        return prefix+s

    skip_layers = []
    multiscale_layers = []
    multiscale_factors = []

    inp = tf.keras.layers.Input(input_shape)

    layer = inp
    
    # down ...
    for n in range(n_depth):
        layer = resnet_block(n_filter_base * 2 ** n,
                             (kernel_size,)*2,
                             n_conv_per_block=n_conv_per_depth,
                             activation=activation,
                             batch_norm=batch_norm)(layer)
            
        skip_layers.append(layer)
        layer = pooling(pool_size)(layer)

    heads = []
        
    # ...and up with skip layers
    for n in reversed(range(n_depth)):
        layer = resnet_block(n_filter_base * 2 ** n,
                             (kernel_size,)*2,
                             n_conv_per_block=n_conv_per_depth,
                             activation=activation,
                             batch_norm=batch_norm)(layer)
            
        # the multi heads
        multi = resnet_block(n_filter_base,
                             (kernel_size,)*2,
                             n_conv_per_block=n_conv_per_depth,
                             activation=activation,
                             batch_norm=batch_norm)(layer)
        
        multi = conv_block(n_channel_out,1,1,activation=last_activation, dtype='float32', name =_name("multi_%s" %n))(multi)
        
        multiscale_layers.append(multi)
        multiscale_factors.append(pool_size**(n+1))
        
        layer = upsampling(pool_size)(layer)            
        layer = tf.keras.layers.Concatenate()([layer, skip_layers[n]])


    final = conv_block(n_channel_out, 1,1, activation=last_activation, dtype='float32', name=_name("final"))(layer)
    multiscale_factors.append(1)
    
    multiscale_layers = multiscale_layers[::-1]
    multiscale_factors = multiscale_factors[::-1]
    
    return tf.keras.models.Model(inp, [final]+multiscale_layers), tuple(multiscale_factors)






    
