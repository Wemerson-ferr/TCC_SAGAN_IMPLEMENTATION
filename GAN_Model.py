import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import numpy as np

from generator import Generator
from discriminator import Discriminator

class GAN(Model):
    """
    Representa a arquitetura GAN que combina o Gerador e o Discriminador.
    """
    def __init__(self, z_dim, gf_dim, df_dim, **kwargs):  # Removido num_classes
        super(GAN, self).__init__(**kwargs)
        
        # As classes Generator e Discriminator agora não precisam mais de 'num_classes'
        self.generator = Generator(gf_dim, z_dim)
        self.discriminator = Discriminator(df_dim)
        self.z_dim = z_dim
        # Removido self.num_classes

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.d_metric = tf.keras.metrics.Mean(name='d_loss')
        self.g_metric = tf.keras.metrics.Mean(name='g_loss')
    
    @property
    def metrics(self):
        return [self.d_metric, self.g_metric]

    def train_step(self, data):
        # O 'data' agora contém apenas as imagens reais, sem os rótulos
        real_images = data

        batch_size = tf.shape(real_images)[0]
        random_z = tf.random.normal(shape=(batch_size, self.z_dim))
        
        # Não há necessidade de gerar rótulos falsos
        
        with tf.GradientTape() as tape:
            # O gerador agora recebe apenas o vetor de ruído
            fake_images = self.generator(random_z)
            # O discriminador agora recebe apenas a imagem
            real_output = self.discriminator(real_images)
            fake_output = self.discriminator(fake_images)
            d_loss = self.d_loss_fn(real_output, fake_output)

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.d_metric.update_state(d_loss)

        with tf.GradientTape() as tape:
            random_z = tf.random.normal(shape=(batch_size, self.z_dim))
            
            # O gerador agora recebe apenas o vetor de ruído
            fake_images = self.generator(random_z)
            # O discriminador agora recebe apenas a imagem
            fake_output = self.discriminator(fake_images)
            g_loss = self.g_loss_fn(fake_output)

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.g_metric.update_state(g_loss)

        return {"d_loss": self.d_metric.result(), "g_loss": self.g_metric.result()}

# As funções de perda permanecem as mesmas
def hinge_d_loss_fn(real_output, fake_output):
    d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - real_output))
    d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + fake_output))
    return d_loss_real + d_loss_fake

def hinge_g_loss_fn(fake_output):
    return -tf.reduce_mean(fake_output)
