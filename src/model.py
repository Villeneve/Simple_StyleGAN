import keras
import keras.layers as lay
import tensorflow as tf
# from src.utils import apply_spectral_normalization_to_model

@keras.saving.register_keras_serializable()
class AdaIN(lay.Layer):
    def __init__(self, channels, **kargs):
        super().__init__(**kargs)
        self.channels = channels
        self.dense_gamma = lay.Dense(channels,name='Dense_Gamma')
        self.dense_bias = lay.Dense(channels,name='Dense_Bias')

    def call(self, inputs, *args, **kwargs):
        features_map, style_w = inputs
        gamma = self.dense_gamma(style_w)
        bias = self.dense_bias(style_w)
        gamma = tf.reshape(gamma,(-1,1,1,self.channels))
        bias = tf.reshape(bias,(-1,1,1,self.channels))
        mean, variance = tf.nn.moments(features_map,(1,2),keepdims=True)
        normalized = tf.nn.batch_normalization(features_map,mean,variance,bias,gamma,1e-6)
        return normalized
    
@keras.saving.register_keras_serializable()
class StyleGAN(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        # Neurônios de conversão de z para w
        self.mapping_network = keras.Sequential([
            lay.InputLayer(shape=(512,)),
            *[lay.Dense(256,activation='leaky_relu') for i in range(8)]
            ],name='mapping_network')

        # Mapa inicial
        self.latent_map = self.add_weight(
            (1,4,4,512),
            trainable=True
        )

        # Convoluções para super-resolução
        self.conv2d = [
            lay.Conv2D(512,3,1,'same',activation='leaky_relu'),
            lay.Conv2D(256,3,1,'same',activation='leaky_relu'),
            lay.Conv2D(256,3,1,'same',activation='leaky_relu'),
            lay.Conv2D(128,3,1,'same',activation='leaky_relu'),
            lay.Conv2D(128,3,1,'same',activation='leaky_relu'),
            lay.Conv2D(64,3,1,'same',activation='leaky_relu'),
            lay.Conv2D(64,3,1,'same',activation='leaky_relu'),
            lay.Conv2D(3,1,1,'same',activation='tanh',name='toRGB')
        ]
        
        # Escaladores de ruído
        self.noises = [
            self.add_weight(shape=(1,1,1,512),trainable=True),
            self.add_weight(shape=(1,1,1,512),trainable=True),
            self.add_weight(shape=(1,1,1,256),trainable=True),
            self.add_weight(shape=(1,1,1,256),trainable=True),
            self.add_weight(shape=(1,1,1,128),trainable=True),
            self.add_weight(shape=(1,1,1,128),trainable=True),
            self.add_weight(shape=(1,1,1,64),trainable=True),
            self.add_weight(shape=(1,1,1,64),trainable=True),
        ]

        # Camadas de Normalização de instância Adaptativa
        self.AdaIN = [
            AdaIN(512),
            AdaIN(512),
            AdaIN(256),
            AdaIN(256),
            AdaIN(128),
            AdaIN(128),
            AdaIN(64),
            AdaIN(64),
        ]

        # Camadas de upsampling
        self.upsampling2d = [lay.UpSampling2D() for i in range(3)]

    # Quando o modelo for chamado para inferência ou treinamento, faz:
    def call(self, z, *args, **kwargs):

        batch_size = z.shape[0]
        w = self.mapping_network(z)
        initial_map = tf.tile(self.latent_map,(batch_size,1,1,1))

        # Bloco do 4x4
        x = initial_map + self.noises[0]*tf.random.normal((batch_size,4,4,1))
        x = self.AdaIN[0]([x,w])
        x = self.conv2d[0](x) + self.noises[1]*tf.random.normal((batch_size,4,4,1))
        x = self.AdaIN[1]([x,w])

        # Bloco do 8x8
        x = self.upsampling2d[0](x)
        x = self.conv2d[1](x) + self.noises[2]*tf.random.normal((batch_size,8,8,1))
        x = self.AdaIN[2]([x,w])
        x = self.conv2d[2](x) + self.noises[3]*tf.random.normal((batch_size,8,8,1))
        x = self.AdaIN[3]([x,w])

        # Bloco do 16x16
        x = self.upsampling2d[1](x)
        x = self.conv2d[3](x) + self.noises[4]*tf.random.normal((batch_size,16,16,1))
        x = self.AdaIN[4]([x,w])
        x = self.conv2d[4](x) + self.noises[5]*tf.random.normal((batch_size,16,16,1))
        x = self.AdaIN[5]([x,w])

        # Bloco do 32x32
        x = self.upsampling2d[2](x)
        x = self.conv2d[5](x) + self.noises[6]*tf.random.normal((batch_size,32,32,1))
        x = self.AdaIN[6]([x,w])
        x = self.conv2d[6](x) + self.noises[7]*tf.random.normal((batch_size,32,32,1))
        x = self.AdaIN[7]([x,w])

        x = self.conv2d[-1](x)

        return x

def create_discriminator():
    inputs = lay.Input((32,32,3))
    model = keras.applications.VGG19(include_top=False,input_tensor=inputs,weights=None)
    x = lay.Flatten()(model.output)
    x = lay.Dense(1,activation='sigmoid')(x)
    discriminator = keras.Model(inputs,x)
    # discriminator = apply_spectral_normalization_to_model(discriminator)
    return discriminator

def create_opt():
    return [keras.optimizers.Adam(1e-4,0.0,.99), keras.optimizers.Adam(1e-4,0.0,.99)]