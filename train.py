import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import gc
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
cars = load_dataset(batch_size)

# Optimizers
opt = create_opt()

# Epochs
epochs = 10000

t = time.time()

# Tensor for epochs counter (DO NOT USE A PYTHON VARIABLE!)
count = tf.Variable(0,trainable=False)

for epoch in range(epochs):

    # Loop for all batchs
    for batch in tqdm(cars, desc=f"Epoch {epoch}/{epochs}", unit="batch"):
        loss = train_step(gan,batch,opt,count)

    # Tensor epochs counter increment
    count.assign_add(1)

    # Print loss
    if epoch % 10 == 0:
        print(f'{epoch} - G = {loss[0]:.4f}; D = {loss[1]:.4f}; Time = {((time.time()-t)):.2f} s')
        t = time.time()

    # Plot 10x10 images each 50 epochs
    if epoch % 50 == 0:
        n = 10
        img = gan[0](tf.random.normal((n**2,512)),training=False)
        fig, ax = plt.subplots(n,n,figsize=(7,7))
        ax = ax.ravel()
        for ii in range(n**2):
            ax[ii].matshow(np.uint8(img[ii]*127.5+127.5))
            ax[ii].set_axis_off()
        plt.tight_layout(pad=0)
        plt.savefig(f'src/imgs/fig_{epoch}.png')
        plt.close()
        del img
        generator.save('src/models/style.keras')
        gc.collect()
