import tensorflow as tf
import keras
import keras.layers as lay

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

bce = keras.losses.BinaryCrossentropy(from_logits=True)

@tf.function
def train_step(gan,batch,opt,epoch):
    g_loss = 0.
    d_loss = 0.
    batch_size = tf.shape(batch)[0]
    

    with tf.GradientTape() as g_tape:
        latent_z = tf.random.normal((batch_size*2,256))
        fake_imgs = gan[0](latent_z, training=True)
        fake_logits = gan[1](fake_imgs, training=True)
        g_loss = bce(tf.ones_like(fake_logits),fake_logits)
        
        
        
    with tf.GradientTape() as d_tape:
        latent_z = tf.random.normal((batch_size,256))
        fake_imgs = gan[0](latent_z, training=True)
        true_logis = gan[1](batch,training=True)
        fake_logits = gan[1](fake_imgs, training=True)
        d_loss = bce(tf.ones_like(true_logis),true_logis)+bce(tf.zeros_like(fake_logits),fake_logits)
        def isTrue():
            reg = r1_regularization(gan[1],batch,10)# + gradient_penalty(gan,batch)
            return reg
        def isFalse():
            return 0.
        d_loss += tf.cond(tf.equal(epoch,True),isTrue,isFalse)

    g_grads = g_tape.gradient(g_loss,gan[0].trainable_variables)
    opt[0].apply_gradients(zip(g_grads,gan[0].trainable_variables))

    d_grads = d_tape.gradient(d_loss,gan[1].trainable_variables)
    opt[1].apply_gradients(zip(d_grads,gan[1].trainable_variables))

    return g_loss, d_loss