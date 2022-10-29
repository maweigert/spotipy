from csbdeep.internals.blocks import conv_block2, resnet_block
from csbdeep.utils import backend_channels_last
import tensorflow as tf
from typing import Tuple


def fpn_resnet(
    input_shape: Tuple[int, int],
    n_channel_out: int=1,
    n_depth: int=3,
    n_filter_base: int=32,
    kernel_size: int=3,
    last_activation: str="lineaer",
    n_conv_per_depth: int=2,
    activation: str="relu",
    batch_norm: bool=False,
    pool_size: int=2,
    kernel_init: str="he_normal",
    prefix: str=""
):
    if last_activation is None:
        last_activation = activation

    def _name(s):
        return prefix+s

    feature_maps = []

    inp = tf.keras.layers.Input(input_shape)
    layer = inp

    # Encoder
    for n in range(n_depth):
        layer = resnet_block(n_filter_base * 2**n,
                             (kernel_size,)*2,
                             n_conv_per_block=n_conv_per_depth,
                             activation=activation,
                             batch_norm=batch_norm,
                             kernel_initializer=kernel_init,
                            )(layer)
        feature_maps.append(layer)

    # FPN module

    fpn_outputs = [None]*len(feature_maps)

    fpn_outputs[-1] = conv_block2(
        n_channel_out,
        1,1,
        name=_name(f"fpn_lv{n_depth}"),
        activation="linear",
        init=kernel_init,
    )(feature_maps[-1])
    upsampled = [None]*len(feature_maps)
    for lv in reversed(range(0, len(feature_maps)-1)):
        # Upsample previous FPN output
        upsampled = tf.keras.layers.UpSampling2D((pool_size,)*2, interpolation="bilinear")(fpn_outputs[lv+1])
        # Upsampled + 1x1 conv
        fpn_outputs[lv] = upsampled + conv_block2(
            n_channel_out,
            1, 1,
            activation="linear",
            name=_name(f"fpn_lv{lv}"),
            init=kernel_init,
        )(feature_maps[lv])
    if last_activation == "sigmoid":
        fpn_outputs = [tf.keras.activations.sigmoid(f) for f in fpn_outputs]
    elif last_activation != "linear":
        raise NotImplementedError("Only sigmoid and linear activations in the output FPN feature maps are currently supported.")
    return tf.keras.models.Model(inp, fpn_outputs), tuple([pool_size**n for n in range(n_depth)])

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
    multiscale_factors.append(1)
    
    multiscale_layers = multiscale_layers[::-1]
    multiscale_factors = multiscale_factors[::-1]
    return tf.keras.models.Model(inp, [final]+multiscale_layers), tuple(multiscale_factors)




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






    
