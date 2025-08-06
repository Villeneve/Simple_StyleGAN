import tensorflow as tf
import keras

@tf.function
def r1_regularization(discriminator, batch):
    with tf.GradientTape() as tape:
        tape.watch(batch)
        logits = discriminator(batch,training=True)
        logits = tf.reduce_sum(logits)
    grads = tape.gradient(logits,[batch])[0]
    norm = tf.reduce_mean(tf.reduce_sum(tf.square(grads),axis=[1,2,3]))
    del grads
    return norm

@tf.function
def train_step(gan,batch,opt,batch_size):
    g_loss = 0.
    d_loss = 0.
    latent_z = tf.random.normal((batch_size,128))
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_imgs = gan[0](latent_z, trainable=True)
        true_logis = gan[1](batch,trainable=True)
        fake_logits = gan[1](fake_imgs, trainable=True)
        g_loss = keras.losses.binary_crossentropy(tf.ones_like(fake_logits), fake_logits)# - 1e-4*tf.reduce_sum(tf.math.reduce_std(fake_imgs,axis=0))
        d_loss = keras.losses.binary_crossentropy(tf.ones_like(true_logis),true_logis)+keras.losses.binary_crossentropy(tf.zeros_like(fake_logits),fake_logits)
        d_loss += r1_regularization(gan[1],batch)

    g_grads = g_tape.gradient(g_loss,gan[0].trainable_variables)
    opt[0].apply_gradients(zip(g_grads,gan[0].trainable_variables))

    d_grads = d_tape.gradient(d_loss,gan[1].trainable_variables)
    opt[1].apply_gradients(zip(d_grads,gan[1].trainable_variables))
    
    return tf.reduce_mean(g_loss), tf.reduce_mean(d_loss)

