import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

sys.path.append('utils')
from nets_V2 import *
from datas_V2 import *

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

# for test
def sample_y(m, n, ind):
    y = np.zeros([m, n])
    for i in range(m):
        y[i, ind] = 1
    return y

def concat(z, y):
    return tf.concat([z, y], 1)

class CGAN():
    def __init__(self, generator, discriminator, data):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        # data
        self.z_dim = self.data.z_dim
        self.y_dim = self.data.y_dim  # condition
        self.X_dim = self.data.X_dim

        # define optimizer
        self.generator_optimizer = tf.keras.optimizers.Adam()
        self.discriminator_optimizer = tf.keras.optimizers.Adam()
        
        # Checkpointing
        self.checkpoint_dir = './ckpt'
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_dir, max_to_keep=3)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=real_output, labels=tf.ones_like(real_output)))
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_output, labels=tf.zeros_like(fake_output)))
        return real_loss + fake_loss

    def generator_loss(self, fake_output):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_output, labels=tf.ones_like(fake_output)))

    @tf.function
    def train_step(self, X_b, y_b):
        z = sample_z(X_b.shape[0], self.z_dim)
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images
            G_sample = self.generator(concat(z, y_b), training=True)
            
            # Get discriminator outputs
            D_real, _ = self.discriminator(concat(X_b, y_b), training=True)
            D_fake, _ = self.discriminator(concat(G_sample, y_b), training=True)
            
            # Calculate losses
            gen_loss = self.generator_loss(D_fake)
            disc_loss = self.discriminator_loss(D_real, D_fake)
            
        # Calculate gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss
        
    def train(self, sample_dir, ckpt_dir='ckpt', training_epoches=100000, batch_size=100):
        fig_count = 0
        
        for epoch in range(training_epoches):
            # Get batch data
            X_b, y_b = self.data(batch_size)
            X_b = tf.convert_to_tensor(X_b, dtype=tf.float32)
            y_b = tf.convert_to_tensor(y_b, dtype=tf.float32)
            
            # Train step
            G_loss_curr, D_loss_curr = self.train_step(X_b, y_b)
            
            # Print loss and save samples
            if epoch % 100 == 0 or epoch < 100:
                print(f'Iter: {epoch}; D loss: {D_loss_curr:.4f}; G_loss: {G_loss_curr:.4f}')

                if epoch % 1000 == 0:
                    y_s = sample_y(16, self.y_dim, fig_count % 10)
                    y_s = tf.convert_to_tensor(y_s, dtype=tf.float32)
                    z = sample_z(16, self.z_dim)
                    z = tf.convert_to_tensor(z, dtype=tf.float32)
                    
                    samples = self.generator(concat(z, y_s), training=False)
                    samples = samples.numpy()
                    
                    fig = self.data.data2fig(samples)
                    plt.savefig(f'{sample_dir}/{str(fig_count).zfill(3)}_{str(fig_count % 10)}.png', bbox_inches='tight')
                    fig_count += 1
                    plt.close(fig)

                if epoch % 2000 == 0:
                    self.manager.save()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # save generated images
    sample_dir = 'Samples/mnist_cgan_mlp_new1'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # param
    generator = G_mlp_mnist()
    discriminator = D_mlp_mnist()

    data = mnist('mlp')

    # run
    cgan = CGAN(generator, discriminator, data)
    cgan.train(sample_dir)