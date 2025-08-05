import tensorflow as tf

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