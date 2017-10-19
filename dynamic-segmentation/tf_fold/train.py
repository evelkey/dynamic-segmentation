# coding: utf-8

# # Dynet segmentation with tf fold


#just a bunch of fun
import numpy as np
import six
import time
from multiprocessing import Process, Queue
import time
import data
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
import tensorflow_fold as td
from conv_lstm_cell import *

# params
BATCH_SIZE = 8
data_dir = "/mnt/permanent/Home/nessie/velkey/data/"

#our alphabet

vocabulary = data.vocabulary(data_dir + 'vocabulary')
vsize=len(vocabulary)
print(vocabulary)

index = lambda char: vocabulary.index(char)
char = lambda i: vocabulary[i]


class data():
    def __init__(self, folder, truncate=500):
        self.data_dir = folder
        self.data = dict()
        self.size = dict()
        self.datasets = ["train", "test", "validation"]
        self.truncate = truncate
        
        for dataset in self.datasets:
            self.data[dataset] = self.sentence_reader(folder+dataset)
            #self.size[dataset] = sum(1 for line in open(folder+dataset))

                        
    def sentence_reader(self, file):
        """
        read sentences from the data format setence: word\tword\n.....\t\n
        """
        data = [line[:-1].split('\t') for line in open(file)]
        for item in data:
            tags = [int(num) for num in item[1]]
            if len(item[0]) == len(tags) and len(tags) != 0:
                sent_onehot = self.onehot(item[0])
                if len(sent_onehot) >= self.truncate:
                    sent_onehot=sent_onehot[:self.truncate]
                    tags = tags[:self.truncate]
                yield (sent_onehot, tags)    
          
            
    def onehot(self, string):
        onehot = np.zeros([len(string),vsize])
        indices = np.arange(len(string)), np.array([int(index(char)) for char in string])
        onehot[indices]=1
        return [onehot[i,:] for i in range(len(onehot))]
            
store = data("/mnt/permanent/Home/nessie/velkey/data/")


# ## helper functions

def params_info():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(variable.name, shape)
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
        print("\tparams: ", variable_parametes)
        total_parameters += variable_parametes
    print(total_parameters)
    return total_parameters

def onehot(string):
    onehot = np.zeros([len(string),vsize])
    onehot[np.arange(len(string)), np.array([index(char) for char in string])]=1
    return [onehot[i,:] for i in range(len(onehot))]

def convLSTM_cell(kernel_size, out_features = 64):
    convlstm = Conv1DLSTMCell(input_shape=[vsize,1], output_channels=out_features, kernel_shape=[kernel_size])
    return td.ScopedLayer(convlstm)

def multi_convLSTM_cell(kernel_sizes, out_features):
    stacked_convLSTM = tf.contrib.rnn.MultiRNNCell()
    return td.ScopedLayer(stacked_convLSTM)

def FC_cell(units):
    return td.ScopedLayer(tf.contrib.rnn.LSTMCell(num_units=units))

def multi_FC_cell(units_list):
    return td.ScopedLayer(tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_units=units) for units in units_list]))
    
def bidirectional_dynamic_CONV(fw_cell, bw_cell, out_features=64):
    bidir_conv_lstm = td.Composition()
    with bidir_conv_lstm.scope():        
        fw_seq = td.Identity().reads(bidir_conv_lstm.input)
        bw_seq = td.Slice(step=-1).reads(fw_seq)

        forward_dir = (td.RNN(fw_cell) >> td.GetItem(0)).reads(fw_seq)
        back_dir = (td.RNN(bw_cell) >> td.GetItem(0)).reads(bw_seq)
        back_to_leftright = td.Slice(step=-1).reads(back_dir)
        
        output_transform = (td.Function(lambda x: tf.reshape(x, [-1,vsize*out_features])) >>
                            #td.FC(10, activation=tf.nn.tanh) >>
                            td.FC(1, activation=None))
        
        bidir_common = (td.ZipWith(td.Concat() >> 
                                  output_transform >> 
                                  td.Metric('logits'))).reads(forward_dir, back_to_leftright)
                    
        #tag_logits = td.Map(output_transform).reads(bidir_common)

        bidir_conv_lstm.output.reads(bidir_common)
    return bidir_conv_lstm

def bidirectional_dynamic_FC(fw_cell, bw_cell, hidden):
    bidir_conv_lstm = td.Composition()
    with bidir_conv_lstm.scope():        
        fw_seq = td.Identity().reads(bidir_conv_lstm.input)
        bw_seq = td.Slice(step=-1).reads(fw_seq)

        forward_dir = (td.RNN(fw_cell) >> td.GetItem(0)).reads(fw_seq)
        back_dir = (td.RNN(bw_cell) >> td.GetItem(0)).reads(bw_seq)
        back_to_leftright = td.Slice(step=-1).reads(back_dir)
        
        output_transform = td.FC(1, activation=tf.nn.sigmoid)
        
        bidir_common = (td.ZipWith(td.Concat() >> 
                                  output_transform >> 
                                  td.Metric('logits'))).reads(forward_dir, back_to_leftright)

        bidir_conv_lstm.output.reads(bidir_common)
    return bidir_conv_lstm


CONV_data = td.Map(td.Vector(vsize) >> td.Function(lambda x: tf.reshape(x, [-1,vsize,1])))
CONV_model =  CONV_data >> bidirectional_dynamic_CONV(convLSTM_cell(vsize), convLSTM_cell(vsize)) >> td.Void()
#labels = td.Map(td.Scalar() >> td.Metric("labels")) >> td.Void()

FC_data = td.Map(td.Vector(vsize))#>> td.Function(lambda x: tf.reshape(x, [-1,vsize])))
FC_model = FC_data >> bidirectional_dynamic_FC(multi_FC_cell([100]), multi_FC_cell([100]),100) >>td.Void()







def bidirectional_dynamic_FC(fw_cell, bw_cell, hidden):
    bidir_conv_lstm = td.Composition()
    with bidir_conv_lstm.scope():        
        fw_seq = td.Identity().reads(bidir_conv_lstm.input[0])
        labels = (td.GetItem(1)>>td.Map(td.Metric("labels"))>>td.Void()).reads(bidir_conv_lstm.input)
        bw_seq = td.Slice(step=-1).reads(fw_seq)

        forward_dir = (td.RNN(fw_cell) >> td.GetItem(0)).reads(fw_seq)
        back_dir = (td.RNN(bw_cell) >> td.GetItem(0)).reads(bw_seq)
        back_to_leftright = td.Slice(step=-1).reads(back_dir)
        
        output_transform = td.FC(1, activation=tf.nn.sigmoid)
        
        bidir_common = (td.ZipWith(td.Concat() >> 
                                  output_transform >> td.Metric('logits'))).reads(forward_dir, back_to_leftright)
        
        bidir_conv_lstm.output.reads(bidir_common)
    return bidir_conv_lstm

d = td.Record((td.Map(td.Vector(vsize)),td.Map(td.Scalar())))
f = d >> bidirectional_dynamic_FC(multi_FC_cell([1000,1000,1000,1000,1000]), multi_FC_cell([1000,1000,1000,1000,1000]),1000) >>td.Void()




compiler = td.Compiler.create(f)
logits = tf.squeeze(compiler.metric_tensors['logits'])
labels = compiler.metric_tensors['labels']

l1_loss = tf.reduce_mean(tf.abs(tf.subtract(labels,logits)))
l2_loss = tf.reduce_mean(tf.abs(tf.subtract(labels,logits)))
cross_entropy_tf = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
cross_entropy =  tf.reduce_mean(labels * - tf.log(logits + 0.00001) + (1 - labels) * - tf.log(1 - logits - 0.00001))
#TODO data label distribution analysis for determining the best loss

loss = cross_entropy

path = "/mnt/permanent/Home/nessie/velkey/logs/cross_entropy_b8_c5x1000_v0"
writer = tf.summary.FileWriter(path, graph=tf.get_default_graph())
saver = tf.train.Saver()
tf.summary.scalar("batch_loss", loss)
acc = tf.reduce_sum(tf.cast(tf.equal(tf.less(0.5,logits), tf.cast(labels, tf.bool)),tf.int32))*100/tf.size(logits)
tf.summary.scalar("batch_accuracy", acc)

                  
summary_op = tf.summary.merge_all()


opt = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = opt.minimize(loss)
sess.run(tf.global_variables_initializer())


sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
#feed = compiler.build_feed_dict([(onehot("naGon jó ötlet"),[0,0,0,1,0,1,0,0,0,0,0,1,0,0]) for i in range(1)])C
#feed = compiler.build_feed_dict([next(store.data["train"]) for _ in range(BATCH_SIZE)])
"""
for i in range(54000):
    a,b,c,d, accu, summ = sess.run([logits, labels, loss, train_op, acc, summary_op], compiler.build_feed_dict([next(store.data["train"]) for _ in range(BATCH_SIZE)]))
    writer.add_summary(summ,i)
    assert not np.isnan(c)
    print("step: ", i)
    print("preds: ", a)
    print("labels: ", b)
    print("loss: ", c, '\n')
    print("accuracy: ", accu, "%")
    
    if i % 1000:
        save_path = saver.save(sess, path, global_step=i)
"""
num_epochs = 10

train_set = compiler.build_loom_inputs(store.data["train"])
dev_feed_dict = compiler.build_feed_dict(store.data["validation"])
test_feed_dict = compiler.build_feed_dict(store.data["test"])
train_feed_dict = {}
i=0
for epoch, shuffled in enumerate(td.epochs(train_set, num_epochs), 1):
    train_loss = 0.0
    for batch in td.group_by_batches(shuffled, BATCH_SIZE):
        train_feed_dict[compiler.loom_input_tensor] = batch
        _, batch_loss, summary = sess.run([train_op, loss, summary_op], train_feed_dict)
        #print(batch_#loss)
        assert not np.isnan(batch_loss)
        writer.add_summary(summary, i)
        i += 1
        train_loss += batch_loss
    dev_loss = sess.run([loss], dev_feed_dict)
    print("DEV loss: ", dev_loss)
    save_path = saver.save(sess, path, global_step=i)
    