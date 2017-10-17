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
EMBEDDING_SIZE = 64
SEP = "|"
BATCH_SIZE = 100
data_dir = "/mnt/permanent/Home/nessie/velkey/data/"

#our alphabet

vocabulary = data.vocabulary(data_dir + 'vocabulary')
vsize=len(vocabulary)
print(vocabulary)

index = lambda char: vocabulary.index(char)
char = lambda i: vocabulary[i]




class data():
    def __init__(self, folder, truncate=100):
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
        while True:
            i=0
            sentence = []
            end_sentence = False
            with open(file) as f:
                for lines in f:
                    line = lines[:-1].split('\t')
                    if line[0] != "":
                        sentence.append(line)
                    else:
                        end_sentence = True
                        i+=1
                    if end_sentence:
                        end_sentence = False
                        sent = " ".join([word[0] for word in sentence])
                        segmented = " ".join([word[1].replace(" ","|") for word in sentence])
                        tags = []
                        last_char = "_"
                        for char in segmented:
                            if char != "|":
                                tags.append(0 if last_char!="|" else 1)
                            last_char = char
                        if len(sent) != 0:
                            sent_onehot = self.onehot(sent)
                            if len(sent_onehot) == len(tags):
                                if len(sent_onehot) >= self.truncate:
                                    sent_onehot=sent_onehot[:self.truncate]
                                    tags = tags[:self.truncate]
                                yield (sent_onehot, tags)
                            sentence = []      
          
            
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


data = td.Map(td.Vector(vsize) >> td.Function(lambda x: tf.reshape(x, [-1,vsize,1])))
model =  data >> bidirectional_dynamic_CONV(convLSTM_cell(vsize), convLSTM_cell(vsize)) >> td.Void()
labels = td.Map(td.Scalar() >> td.Metric("labels")) >> td.Void()

FC_data = td.Map(td.Vector(vsize))#>> td.Function(lambda x: tf.reshape(x, [-1,vsize])))
FC_model = FC_data >> bidirectional_dynamic_FC(multi_FC_cell([100]), multi_FC_cell([100]),100) >>td.Void()



compiler = td.Compiler.create((FC_model, labels))
logits = tf.squeeze(compiler.metric_tensors['logits'])
labels = compiler.metric_tensors['labels']

loss = tf.reduce_mean(tf.abs(tf.subtract(labels,logits)))
l2_loss = tf.reduce_mean(tf.abs(tf.subtract(labels,logits)))
log_loss = (labels) * tf.log(logits) + (1 - labels) * tf.log(1 - logits)
#cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
best_loss =  labels * -tf.log(logits) + (1 - labels) * -tf.log(1 - logits)
#TODO data label distribution analysis for determining the better best lost ;)


opt = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = opt.minimize(best_loss)
sess.run(tf.global_variables_initializer())



sess.run(tf.global_variables_initializer())
#feed = compiler.build_feed_dict([(onehot("naGon jó ötlet"),[0,0,0,1,0,1,0,0,0,0,0,1,0,0]) for i in range(1)])C
x = next(store.data["train"])
x = next(store.data["train"])
print(x)
exit()
feed = compiler.build_feed_dict([x for _ in range(BATCH_SIZE)])
for i in range(100000):
    a,b,c,d= sess.run([logits, compiler.metric_tensors['labels'], loss, train_op], feed) #compiler.build_feed_dict([next(store.data["train"]) for _ in range(BATCH_SIZE)]))
    print("step: ", i)
    print("preds: ", a)
    print("labels: ", b)
    print("loss: ", c, '\n')
    accuracy=a > 0.5
    print("accuracy: ", np.sum(np.equal(accuracy,b))/len(accuracy)*100, "%")

