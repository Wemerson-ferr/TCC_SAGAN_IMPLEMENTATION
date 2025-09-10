import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dense, Reshape, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow_addons.layers import SpectralNormalization

def usample(x, size):
    """Upscales o vetor de entrada por um fator de 2 usando vizinho mais próximo."""
    return tf.image.resize(x, [size[0] * 2, size[1] * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

class ConditionalBatchNorm(Layer):
    def __init__(self, num_features, num_classes, **kwargs):
        super(ConditionalBatchNorm, self).__init__(**kwargs)
        self.num_features = num_features
        self.num_classes = num_classes
        self.bn = BatchNormalization(center=False, scale=False)
        self.embed_gamma = Dense(num_features, use_bias=False)
        self.embed_beta = Dense(num_features, use_bias=False)

    def call(self, inputs, labels):
        gamma = self.embed_gamma(labels)
        beta = self.embed_beta(labels)
        gamma = tf.reshape(gamma, [-1, 1, 1, self.num_features])
        beta = tf.reshape(beta, [-1, 1, 1, self.num_features])
        normalized = self.bn(inputs)
        return normalized * (1 + gamma) + beta

    def get_config(self):
        config = super(ConditionalBatchNorm, self).get_config()
        config.update({
            'num_features': self.num_features,
            'num_classes': self.num_classes,
        })
        return config

class SelfAttention(Layer):
    def __init__(self, channels, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = channels
        self.query_conv = SpectralNormalization(Conv2D(channels // 8, kernel_size=1, padding='same'))
        self.key_conv = SpectralNormalization(Conv2D(channels // 8, kernel_size=1, padding='same'))
        self.value_conv = SpectralNormalization(Conv2D(channels, kernel_size=1, padding='same'))
        self.gamma = tf.Variable(initial_value=tf.zeros(1), trainable=True)
    
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

class GBlock(Layer):
    def __init__(self, out_channels, num_classes, **kwargs):
        super(GBlock, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.num_classes = num_classes

        # Camadas do ramo principal
        self.bn0 = ConditionalBatchNorm(out_channels, num_classes)
        self.conv_main1 = SpectralNormalization(Conv2D(out_channels, 3, padding='same'))
        self.bn1 = ConditionalBatchNorm(out_channels, num_classes)
        self.conv_main2 = SpectralNormalization(Conv2D(out_channels, 3, padding='same'))

        # Camada do atalho (shortcut)
        self.conv_shortcut = SpectralNormalization(Conv2D(out_channels, 1, padding='same'))

    def call(self, inputs, labels):
        x = inputs
        # Upsample e aplicação do atalho
        x_shortcut = usample(x, x.shape[1:3])
        x_shortcut = self.conv_shortcut(x_shortcut)

        # Ramo principal
        x = activations.relu(self.bn0(x, labels))
        x = usample(x, x.shape[1:3])
        x = self.conv_main1(x)
        x = activations.relu(self.bn1(x, labels))
        x = self.conv_main2(x)

        # Adição do atalho
        return x + x_shortcut

    def get_config(self):
        config = super(GBlock, self).get_config()
        config.update({
            'out_channels': self.out_channels,
            'num_classes': self.num_classes,
        })
        return config

class Generator(Model):
    def __init__(self, gf_dim, num_classes, z_dim, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.gf_dim = gf_dim
        self.num_classes = num_classes
        self.z_dim = z_dim
        
        # O modelo começa com uma camada densa e um reshape
        self.dense_in = SpectralNormalization(Dense(gf_dim * 16 * 4 * 4))
        self.reshape = Reshape((4, 4, gf_dim * 16))

        # Os blocos residuais
        self.block1 = GBlock(gf_dim * 16, num_classes) # -> 8x8
        self.block2 = GBlock(gf_dim * 8, num_classes)  # -> 16x16
        self.block3 = GBlock(gf_dim * 4, num_classes)  # -> 32x32

        # Camada de Self-Attention
        self.attention = SelfAttention(gf_dim * 4)

        self.block4 = GBlock(gf_dim * 2, num_classes)  # -> 64x64
        self.block5 = GBlock(gf_dim, num_classes)     # -> 128x128
        
        # Camada de Batch Normalization e a convolução final
        self.final_bn = BatchNormalization()
        self.final_conv = SpectralNormalization(Conv2D(3, 3, padding='same', activation='tanh'))

    def call(self, inputs, labels, training=True):
        
        # Camada de entrada
        x = self.dense_in(inputs)
        x = self.reshape(x)
        
        # Blocos residuais
        x = self.block1(x, labels)
        x = self.block2(x, labels)
        x = self.block3(x, labels)
        
        # Camada de atenção
        x = self.attention(x)
        
        x = self.block4(x, labels)
        x = self.block5(x, labels)
        
        # Camadas finais
        x = activations.relu(self.final_bn(x, training=training))
        x = self.final_conv(x)
        
        return x

    def get_config(self):
        config = super(Generator, self).get_config()
        config.update({
            'gf_dim': self.gf_dim,
            'num_classes': self.num_classes,
            'z_dim': self.z_dim,
        })
        return config
