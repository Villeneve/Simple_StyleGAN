import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import time
import os
# import gc
from tqdm import tqdm

from src.model import *
from src.dataset_loader import *
from src.utils import *

# Create folders
try: os.mkdir('src/imgs/')
except:
    print('imgs folder find')

try: os.mkdir('src/models/')
except:
    print('models folder find')


# Models declaration
generator = StyleGAN()
discriminator = create_discriminator()
gan = [generator, discriminator]

# Dataset load
batch_size = 64
(x,y),(_,_) = keras.datasets.cifar10.load_data()
x = x[y[:,0] == 1]
dataset = tf.data.Dataset.from_tensor_slices((x)).map(lambda x: (tf.cast(x,tf.float32)-127.5)/127.5).cache().shuffle(1000).batch(batch_size,drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# Optimizers
opt = create_opt()

# Epochs
epochs = 999

t = time.time()

# Meadian for loss metric
g_loss, dl_1, dl_2  = [keras.metrics.Mean() for i in range(3)]
g_array_loss = []
d_array_loss = []

for epoch in range(epochs):

    # Loop for all batchs
    for i,batch in enumerate(tqdm(dataset, desc=f"Epoch {epoch}/{epochs}", unit="batch")):
        loss = train_step(gan,batch,opt,i%10==0)
        g_loss.update_state(loss[0])
        dl_1.update_state(loss[1])
        dl_2.update_state(loss[2])
    
    g_array_loss.append(g_loss.result())
    d_array_loss.append((dl_1.result()+dl_2.result())/2)

    g_loss.reset_state()
    dl_1.reset_state()
    dl_2.reset_state()

    # Print loss
    if epoch % 10 == 0:

        plt.plot(g_array_loss,label='G_loss')
        plt.plot(d_array_loss,label='D_loss')
        plt.legend()
        plt.savefig('loss.jpg')
        plt.close()
    

    # Plot 10x10 images each 50 epochs
    if epoch % 50 == 0:
        n = 10
        img = gan[0](tf.random.normal((n**2,256)),training=False)
        fig, ax = plt.subplots(n,n,figsize=(7,7))
        ax = ax.ravel()
        for ii in range(n**2):
            ax[ii].matshow(np.uint8(img[ii]*127.5+127.5))
            ax[ii].set_axis_off()
        plt.tight_layout(pad=0)
        plt.savefig(f'src/imgs/fig_{epoch}.png')
        plt.close()
        generator.save('src/models/style.keras')
