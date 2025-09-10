import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.layers import AveragePooling2D

def dsample(x):
    """Downsamples a tensor by a factor of 2 using average pooling."""
    return AveragePooling2D(pool_size=(2, 2))(x)

class DBlock(Layer):
    """Residual block used in the discriminator."""
    def __init__(self, out_channels, downsample=True, **kwargs):
        super(DBlock, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.downsample = downsample

        self.conv1 = SpectralNormalization(Conv2D(out_channels, 3, padding='same'))
        self.conv2 = SpectralNormalization(Conv2D(out_channels, 3, padding='same'))
        self.conv_shortcut = SpectralNormalization(Conv2D(out_channels, 1, padding='same'))
    
    def call(self, inputs):
        x_shortcut = inputs
        x = activations.relu(inputs)
        x = self.conv1(x)
        x = activations.relu(x)
        x = self.conv2(x)

        if self.downsample:
            x = dsample(x)
            x_shortcut = dsample(x_shortcut)

        if inputs.shape[-1] != self.out_channels:
            x_shortcut = self.conv_shortcut(x_shortcut)

        return x + x_shortcut

class OptimizedDBlock(Layer):
    """Simplified residual block for initial downsampling."""
    def __init__(self, out_channels, **kwargs):
        super(OptimizedDBlock, self).__init__(**kwargs)
        self.out_channels = out_channels

        self.conv1 = SpectralNormalization(Conv2D(out_channels, 3, padding='same'))
        self.conv_shortcut = SpectralNormalization(Conv2D(out_channels, 1, padding='same'))
    
    def call(self, inputs):
        x_shortcut = dsample(inputs)
        x_shortcut = self.conv_shortcut(x_shortcut)

        x = self.conv1(inputs)
        x = activations.relu(x)
        x = SpectralNormalization(Conv2D(self.out_channels, 3, padding='same'))(x)
        x = dsample(x)

        return x + x_shortcut

class SelfAttention(Layer):
    """Self-Attention block for the discriminator."""
    def __init__(self, channels, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = channels
        self.query_conv = SpectralNormalization(Conv2D(channels // 8, kernel_size=1, padding='same'))
        self.key_conv = SpectralNormalization(Conv2D(channels // 8, kernel_size=1, padding='same'))
        self.value_conv = SpectralNormalization(Conv2D(channels, kernel_size=1, padding='same'))
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
    
    def call(self, x):
        batch_size, h, w, c = x.shape
        proj_query = tf.reshape(self.query_conv(x), (batch_size, -1, self.channels // 8))
        proj_key = tf.reshape(self.key_conv(x), (batch_size, -1, self.channels // 8))
        
        energy = tf.matmul(proj_key, proj_query, transpose_b=True)
        attention = tf.nn.softmax(energy)
        
        proj_value = tf.reshape(self.value_conv(x), (batch_size, -1, self.channels))
        
        out = tf.matmul(proj_value, attention, transpose_b=True)
        out = tf.reshape(out, (batch_size, h, w, c))
        
        out = self.gamma * out + x
        return out

class Discriminator(Model):
    """Main Discriminator model."""
    def __init__(self, df_dim, num_classes, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.df_dim = df_dim
        self.num_classes = num_classes

        self.block1 = OptimizedDBlock(df_dim) 
        self.block2 = DBlock(df_dim * 2)
        self.block3 = DBlock(df_dim * 4)
        self.block4 = DBlock(df_dim * 8)
        self.block5 = DBlock(df_dim * 16)
        self.block6 = DBlock(df_dim * 16, downsample=False)

        self.final_dense = SpectralNormalization(Dense(1))
        self.embedding = SpectralNormalization(Dense(df_dim * 16))

    def call(self, inputs, labels, training=True):
        x = inputs
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = activations.relu(x)
        
        # SOMA as features de todas as posições espaciais
        x = tf.reduce_sum(x, axis=[1, 2])
        
        # Classificação real/falso
        logits = self.final_dense(x)

        # Projeção da camada de rótulos
        label_embedding = self.embedding(labels)
        
        # Adiciona a informação de classe aos logits
        logits += tf.reduce_sum(x * label_embedding, axis=1, keepdims=True)

        return logits
