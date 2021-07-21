
import numpy as np
from csbdeep.internals.blocks import conv_block2
from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict, load_json, save_json
import tensorflow as tf


def multiscale_unet(
        input_shape,
        n_channel_out=1,
        n_depth=3,
        n_filter_base=32,
        kernel_size=3,
        last_activation="linear",
        n_conv_per_depth=2,
        activation="relu",
        batch_norm=False,
        dropout=0.0,
        pool_size=2,
        kernel_init="glorot_uniform",
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

        for i in range(n_conv_per_depth):
            layer = conv_block(n_filter_base, kernel_size, kernel_size,
                                   dropout=dropout,
                                   init=kernel_init,
                                   activation=activation,
                                   batch_norm=batch_norm, name=_name("multi_level_%s_no_%s" % (n, i)))(layer)
        multi = conv_block(n_filter_base,3,3,activation=activation, name =_name("multi_pre_%s" %n))(layer)
        multi = conv_block(n_channel_out,1,1,activation=last_activation, name =_name("multi_%s" %n))(multi)

        layer = tf.keras.layers.Concatenate(axis=channel_axis)([
            upsampling(pool_size)(layer), skip_layers[n], upsampling(pool_size)(layer)
        ])

        multiscale_layers.append(multi)
        multiscale_factors.append(pool_size**(n+1))

        for i in range(n_conv_per_depth - 1):
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

    final = conv_block(n_channel_out, 1,1, activation=last_activation, name=_name("final"))(layer)
    multiscale_factors.append(1)
    
    multiscale_layers = multiscale_layers[::-1]
    multiscale_factors = multiscale_factors[::-1]
    
    return tf.keras.models.Model(inp, [final]+multiscale_layers), tuple(multiscale_factors)



if __name__ == '__main__':
    model, scales = multiscale_unet((128,128,1), last_activation="sigmoid")
