import tensorflow as tf
import keras

@tf.function
def r1_regularization(discriminator, batch):
    with tf.GradientTape() as tape:
        tape.watch(batch)
        logits = discriminator(batch,training=False)
        logits = tf.reduce_sum(logits)
    grads = tape.gradient(logits,[batch])[0]
    norm = tf.reduce_mean(tf.reduce_sum(tf.square(grads),axis=[1,2,3])+1e-8)
    del grads
    return norm

@tf.function
def gradient_penalty(gan,batch):
    shape = tf.shape(batch)[0]
    noise = tf.random.normal((shape,128))
    alpha = tf.random.uniform((shape,1,1,1))
    fake_imgs = gan[0](noise,training=False)
    interpolation = fake_imgs + alpha*(batch-fake_imgs)
    with tf.GradientTape() as tape:
        tape.watch(interpolation)
        y = gan[1](interpolation, training=True)
    grads = tape.gradient(y,[interpolation])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads),axis=[1,2,3])+1e-8)
    return tf.reduce_mean(tf.square(norm-1))

@tf.function
def train_step(gan,batch,opt):
    g_loss = 0.
    d_loss = 0.
    batch_size = tf.shape(batch)[0]
    latent_z = tf.random.normal((batch_size,128))

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_imgs = gan[0](latent_z, trainable=True)
        true_logis = gan[1](batch,trainable=True)
        fake_logits = gan[1](fake_imgs, trainable=True)
        g_loss = -tf.reduce_mean(fake_logits)
        d_loss = -(tf.reduce_mean(true_logis) - tf.reduce_mean(fake_logits)) + 10*gradient_penalty(gan,batch)

    g_grads = g_tape.gradient(g_loss,gan[0].trainable_variables)
    opt[0].apply_gradients(zip(g_grads,gan[0].trainable_variables))

    d_grads = d_tape.gradient(d_loss,gan[1].trainable_variables)
    opt[1].apply_gradients(zip(d_grads,gan[1].trainable_variables))
    
    return tf.reduce_mean(g_loss), tf.reduce_mean(d_loss)

