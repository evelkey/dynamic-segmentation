import tensorflow as tf
import numpy as np
from segmentation.conv_lstm_cell import Conv1DLSTMCell

FLAGS = tf.app.flags.FLAGS

def model_information():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(variable.name, shape)
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        print("\tparams: ", variable_parametes)
        total_parameters += variable_parametes
    print(total_parameters)
    return total_parameters


def convolutional_output(input_tensor, outputs, filter_sizes):
    temp = input_tensor
    for out, filt in list(zip(outputs, filter_sizes))[:-1]:
        temp = tf.layers.conv1d(temp, out, filt, activation=tf.nn.sigmoid, padding='same')
    return tf.layers.conv1d(temp, outputs[-1], filter_sizes[-1], activation=None, padding='same')


def stacked_fc_bi_lstm(x, num_units):
    with tf.variable_scope("fw"):
        fw_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=units) for units in num_units])
    with tf.variable_scope("bw"):
        bw_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=units) for units in num_units])
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cells, cell_bw=bw_cells, inputs=x, dtype=tf.float32)

    return tf.concat(outputs, -1)


def stacked_fully_conv_bi_lstm(x, kernel_size, num_units, vsize):
    inputs = tf.expand_dims(x, -1)

    def convLSTM_cell(kernel_size, out_features, in_features, vsize):
        convlstm = Conv1DLSTMCell(input_shape=[vsize, in_features], output_channels=out_features,
                                  kernel_shape=[kernel_size])
        return convlstm

    def multi_convLSTM_cell(kernel_sizes, out_features, vsize):
        in_features = [1] + out_features[1:]
        return tf.contrib.rnn.MultiRNNCell(
            [convLSTM_cell(kernel_sizes[i], out_features[i], in_features[i], vsize)
             for i in range(len(kernel_sizes))])

    with tf.variable_scope("fw"):
        fw_cells = multi_convLSTM_cell([vsize], [1], vsize)
    with tf.variable_scope("bw"):
        bw_cells = multi_convLSTM_cell([vsize], [1], vsize)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cells, cell_bw=bw_cells, inputs=inputs,
                                                      dtype=tf.float32)

    RNN_out = tf.concat(outputs, -1)
    RNN_out = tf.squeeze(RNN_out)

    return tf.reshape(RNN_out, (-1, FLAGS.truncate, vsize * 2))
