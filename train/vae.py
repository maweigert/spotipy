import numpy as np
import tensorflow as tf
from train import get_data
from csbdeep.internals.blocks import unet_block
from scipy.ndimage import gaussian_filter
from augmend import Augmend, Elastic, Identity, FlipRot90, AdditiveNoise, CutOut, Scale, GaussianBlur, Rotate, IntensityScaleShift, IsotropicScale, BaseTransform



class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class VAE(tf.keras.Model):
    def __init__(self, latent_dim=2, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.latent_dim = latent_dim
    
        encoder_inp = tf.keras.layers.Input((None,None,1))
        x = encoder_inp
        x = unet_block(3,32,last_activation='relu')(x)
        z_mean = tf.keras.layers.Conv2D(latent_dim,1,padding='same', activation='linear')(x)
        z_logvar = tf.keras.layers.Conv2D(latent_dim,1,padding='same', activation='linear')(x)
        z = Sampling()([z_mean, z_logvar])
        self.encoder = tf.keras.Model(encoder_inp, [z_mean, z_logvar, z], name="encoder")


        decoder_inp = tf.keras.layers.Input(shape=(None,None, 1))
        decoder_inp_latent = tf.keras.layers.Input(shape=(None,None, latent_dim))
        x = tf.keras.layers.Concatenate(axis=-1)([decoder_inp, decoder_inp_latent])
        x = unet_block(3,32,last_activation='relu')(x)
        out = tf.keras.layers.Conv2D(1,3,padding='same', activation='linear')(x)
        self.decoder = tf.keras.Model([decoder_inp, decoder_inp_latent], out, name="decoder")

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            u = self.decoder([y,z])
            reconstruction_loss = tf.reduce_mean(
                    tf.keras.losses.mae(x*y, u))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = 10*tf.reduce_mean(kl_loss)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


    
if __name__ == '__main__':


    X, Y, P = get_data(folder='train', sigma=4)
    
    aug = Augmend()
    aug.add([FlipRot90(axis = (0,1)),FlipRot90(axis = (0,1))])
    aug.add([Rotate(axis = (0,1)),Rotate(axis = (0,1))])
    t = Elastic(amount=5, grid=10, order=0,axis = (0,1))
    aug.add([t,t])

    X2, Y2 = [], []
    for x,y in zip(X,Y):
        for _ in range(50):
            x2,y2 = aug([x,y])
            X2.append(x2)
            Y2.append(y2)
    
    X = np.expand_dims(np.stack(X2),-1)
    Y = np.expand_dims(np.stack(Y2),-1)
     
    model = VAE()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4))
    model.fit(X,Y, epochs=100, batch_size=16)    
