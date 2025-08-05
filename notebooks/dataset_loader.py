import tensorflow as tf
import keras

def load_dataset(batch_size=64):
    (x,y),(_,_) = keras.datasets.cifar10.load_data()
    x = (tf.cast(x[y[:,0] == 1],tf.float32)-127.5)/127.5
    batch_size = 32
    cars = tf.data.Dataset.from_tensor_slices((x)).shuffle(1000).batch(batch_size,drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return cars