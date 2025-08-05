import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from .src.model import *
from .src.dataset_loader import *
from .src.utils import *


# Models declaration
generator = StyleGAN()
discriminator = create_discriminator()
gan = [generator, discriminator]

# Dataset load
cars = load_dataset(batch_size=64)