import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

from .src.model import *
from .src.dataset_loader import *
from .src.utils import *


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

for epoch in range(epochs):
    t = time.time()
    for batch in cars:
        g_loss, d_loss = train_step(gan,batch,opt,batch_size,epoch)
    if epoch%10 == 0:
        print(f'{epoch} - G = {g_loss:.4f}; D = {d_loss:.4f}; Time = {((time.time()-t)/60.):.2f}')

    if epoch % 50 == 0:
        n = 5
        img = gan[0](tf.random.normal((n**2,128)),training=False)

        fig, ax = plt.subplots(n,n,figsize=(7,7))
        ax = ax.ravel()
        for ii in range(n**2):
            ax[ii].matshow(np.uint8(img[ii]*127.5+127.5))
            ax[ii].set_axis_off()
        plt.tight_layout(pad=0)
        plt.savefig(f'fig_{epoch}.png')
        plt.close()
        del img
        generator.save('style.keras')