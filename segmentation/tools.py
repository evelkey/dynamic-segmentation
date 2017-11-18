import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def metrics(probs, labels, x):
    labels = tf.cast(labels, tf.int32)
    predicted = tf.cast(tf.less(0.5, probs), tf.int32)
    length = tf.reduce_sum(x)

    # crop the sequences:
    labels = labels

    TP = tf.count_nonzero(predicted * labels)
    TN = tf.count_nonzero((predicted - 1) * (labels - 1))
    FP = tf.count_nonzero(predicted * (labels - 1))
    FN = tf.count_nonzero((predicted - 1) * labels)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    accuracy = tf.cast(tf.count_nonzero(tf.equal(predicted, labels)), tf.float32) / tf.cast(tf.size(labels),
                                                                                            tf.float32) * 100

    return precision, recall, accuracy, f1, predicted


def loss(logits, labels):
    if FLAGS.loss == "l1":
        l1_loss = tf.reduce_mean(tf.abs(tf.subtract(logits, tf.cast(labels, tf.float32))))
        return l1_loss
    elif FLAGS.loss == "l2":
        l2_loss = tf.reduce_mean(tf.square(tf.subtract(logits, tf.cast(labels, tf.float32))))
        return l2_loss
    elif FLAGS.loss == "crossentropy":
        cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.float32)))
        return cross_entropy
    else:
        raise NotImplemented


class Stopper:
    def __init__(self, patience=20):
        self.log = []
        self.patience = patience
        self.should_stop = False

    def add(self, value):
        self.log.append(value)
        return self.check()

    def check(self):
        minimum = min(self.log)
        errors = sum([1 if i > minimum else 0 for i in self.log[self.log.index(minimum):]])
        if errors > self.patience:
            self.should_stop = True
        return self.should_stop