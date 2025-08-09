import tensorflow as tf
import keras

@tf.function
def r1_regularization(discriminator, batch, gamma):
    with tf.GradientTape() as tape:
        tape.watch(batch)
        logits = discriminator(batch,training=True)
    grads = tape.gradient(logits,[batch])[0]
    norm = tf.reduce_mean(tf.reduce_sum(tf.square(grads),axis=[1,2,3]))
    return norm*gamma/2

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
def train_step(gan,batch,opt,epoch):
    g_loss = 0.
    d_loss = 0.
    batch_size = tf.shape(batch)[0]
    latent_z = tf.random.normal((batch_size,128))
    bce = keras.losses.BinaryCrossentropy(from_logits=True)

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_imgs = gan[0](latent_z, trainable=True)
        true_logis = gan[1](batch,trainable=True)
        fake_logits = gan[1](fake_imgs, trainable=True)
        g_loss = bce(tf.ones_like(fake_logits),fake_logits,)-tf.reduce_mean(tf.math.reduce_std(fake_imgs,axis=[0,-1]))
        d_loss = bce(tf.ones_like(true_logis),true_logis)+bce(tf.zeros_like(fake_logits),fake_logits)
        def isTrue():
            return d_loss + r1_regularization(gan[1],batch,10)
        def isFalse():
            return d_loss
        d_loss = tf.cond(tf.equal(epoch%10,0),isTrue,isFalse)

    g_grads = g_tape.gradient(g_loss,gan[0].trainable_variables)
    opt[0].apply_gradients(zip(g_grads,gan[0].trainable_variables))

    d_grads = d_tape.gradient(d_loss,gan[1].trainable_variables)
    opt[1].apply_gradients(zip(d_grads,gan[1].trainable_variables))
    
    return tf.reduce_mean(g_loss), tf.reduce_mean(d_loss)

