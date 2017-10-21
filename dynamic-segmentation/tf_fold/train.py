# coding: utf-8
import numpy as np
import time
import data
import tqdm
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
import tensorflow_fold as td
from conv_lstm_cell import *


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 8, """batchsize""")
tf.app.flags.DEFINE_integer('epochs', 2, """epoch count""")
tf.app.flags.DEFINE_string('data_dir', "/mnt/permanent/Home/nessie/velkey/data/", """data store basedir""")
tf.app.flags.DEFINE_string('log_dir', "/mnt/permanent/Home/nessie/velkey/logs/", """logging directory root""")
tf.app.flags.DEFINE_string('run_name', "development", """naming: loss_fn, batch size, 
                                                         architecture, optimizer""")
tf.app.flags.DEFINE_string('data_type', "sentence/", """can be sentence/, word/""")
tf.app.flags.DEFINE_string('model', "lstm", """can be lstm, convlstm right now""")
tf.app.flags.DEFINE_float('learning_rate', 0.1, """starting learning rate""")


vocabulary = data.vocabulary(FLAGS.data_dir + 'vocabulary')
vsize=len(vocabulary)
print(vocabulary)

index = lambda char: vocabulary.index(char)
char = lambda i: vocabulary[i]


class data():
    def __init__(self, folder, truncate=120):
        self.data_dir = folder
        self.data = dict()
        self.size = dict()
        self.datasets = ["train", "test", "validation"]
        self.truncate = truncate
        
        for dataset in self.datasets:
            self.data[dataset] = self.sentence_reader(folder+dataset)
            self.size[dataset] = sum(1 for line in open(folder+dataset))

                        
    def sentence_reader(self, file):
        """
        read sentences from the data format setence: word\tword\n.....\t\n
        """
        data = [line[:-1].split('\t') for line in open(file)]
        while True:
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

store = data(FLAGS.data_dir + FLAGS.data_type)


d = td.Record((td.Map(td.Vector(vsize)),td.Map(td.Scalar())))
f = d >> bidirectional_dynamic_FC(multi_FC_cell([500]*5), multi_FC_cell([500]*5),500) >> td.Void()




compiler = td.Compiler.create(f)
logits = tf.squeeze(compiler.metric_tensors['logits'])
labels = compiler.metric_tensors['labels']

l1_loss = tf.reduce_mean(tf.abs(tf.subtract(labels,logits)))
l2_loss = tf.reduce_mean(tf.abs(tf.subtract(labels,logits)))
cross_entropy_tf = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
cross_entropy =  tf.reduce_mean(labels * - tf.log(logits) + (1 - labels) * - tf.log(1 - logits))
#TODO data label distribution analysis for determining the best loss

loss = cross_entropy

path = FLAGS.log_dir + FLAGS.run_name
writer = tf.summary.FileWriter(path, graph=tf.get_default_graph())
saver = tf.train.Saver()
tf.summary.scalar("batch_loss", loss)

#Accuracy
acc = tf.reduce_sum(tf.cast(tf.equal(tf.less(0.5,logits), tf.cast(labels, tf.bool)),tf.int32))*100/tf.size(logits)
tf.summary.scalar("batch_accuracy", acc)

# Recall
correct_trues = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.less(0.5,logits), tf.cast(labels, tf.bool)), tf.cast(labels, tf.bool)), tf.int32))
all_trues = tf.reduce_sum(labels)
recall = tf.cast(correct_trues,tf.float32) / all_trues
tf.summary.scalar("recall", recall)
         
summary_op = tf.summary.merge_all()
opt = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
train_op = opt.minimize(loss)
sess.run(tf.global_variables_initializer())

#def inference_on_data(sess, data_gen, compiler, ):
#TODO automated validation and test
#TODO Early stopping
#TODO auto save
#TODO word training
#TODO command line usage
#TODO metrics : word level accuracy, char level accuracy, recall F-score
    

for i in tqdm.trange(FLAGS.epochs * int(store.size["train"] / FLAGS.batch_size), unit="batches"):
    _, batch_loss, summary = sess.run([train_op, loss, summary_op], compiler.build_feed_dict([next(store.data["train"]) for _ in range(FLAGS.batch_size)]))
    assert not np.isnan(batch_loss)
    if i % 5 == 0:
        writer.add_summary(summary, i)
    if i % 1000 == 0:
        save_path = saver.save(sess, path, global_step=i)
    

    