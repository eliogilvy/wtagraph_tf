import tensorflow as tf

def evaluate(model, nf, ef, labels, mask):
    n_logits, e_logits = model(nf, ef, training=False)
    e_logits = e_logits[mask]
    labels = labels[mask]
    indices = tf.math.argmax(e_logits, axis=1)
    labels = tf.cast(labels, dtype=tf.int64)
    acc = tf.reduce_sum(tf.cast(indices == labels, dtype=tf.int64))
    return acc.numpy().item() * 1.0 / len(labels)