
# coding: utf-8

# # Dynet segmentation with tf fold
# ![animation](../../fold/tensorflow_fold/g3doc/animation.gif)  

# In[1]:


#just a bunch of fun
import numpy as np
import six
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
data_dir = "/home/moon/data/"

#our alphabet

vocabulary = data.vocabulary(data_dir + 'vocabulary')
vsize=len(vocabulary)
print(vocabulary)

index = lambda char: vocabulary.index(char)
char = lambda i: vocabulary[i]



# In[2]:


class reader():
    def __init__(self, folder, mode="bidirectional", qsize=10000, start=True):
        self.data_dir = folder
        self.qsize= qsize
        self.pool = dict()
        self.data = dict()
        self.datasets = ["train", "test", "validation"]
        if mode == "bidirectional":
            convert_fn = self.bidirectional_sentence_reader
        else:
            convert_fn = self.sentence_reader
            
        for dataset in self.datasets:
            self.data[dataset] = Queue(qsize)
            self.pool[dataset] = Process(target=convert_fn, args=(self.data[dataset], self.data_dir+dataset))
        if start == True:
            self.start()
            
    def sentence_reader(self, queue, file):
        """
        read sentences from the data format setence: word\tword\n.....\t\n
        """
        while True:
            with open(file) as f:
                while True:
                    try:
                        sentence = []
                        while True:
                            line = f.readline()[:-1].split('\t')
                            if line[0] != "":
                                sentence.append(line)
                            else:
                                break
                        sent = " ".join([word[0] for word in sentence])
                        segmented = " ".join([word[1].replace(" ","|") for word in sentence])
                        tags = []
                        last_char = "_"
                        for char in segmented:
                            if char != "|":
                                tags.append(0 if last_char!="|" else 1)
                            last_char = char
                        queue.put((sent, tags))
                    except e:
                        print(e)
                        
    def bidirectional_sentence_reader(self, queue, file):
        """
        read sentences from the data format setence: word\tword\n.....\t\n
        """
        while True:
            with open(file) as f:
                while True:
                    try:
                        sentence = []
                        while True:
                            line = f.readline()[:-1].split('\t')
                            if line[0] != "":
                                sentence.append(line)
                            else:
                                break
                        sent = " ".join([word[0] for word in sentence])
                        segmented = " ".join([word[1].replace(" ","|") for word in sentence])
                        tags = []
                        last_char = "_"
                        for char in segmented:
                            if char != "|":
                                tags.append(0 if last_char!="|" else 1)
                            last_char = char
                        for i in range(1, len(sent)-1):
                            forward, backward = sent[:i], sent[i:][::-1]
                            queue.put(([self.onehot(forward), self.onehot(backward)], tags[i]))
                    except:
                        print("err")
                        
    def start(self):
        for dataset in self.datasets:
            self.pool[dataset].start()
            
    def stop(self):
        for dataset in self.datasets:
            self.pool[dataset].terminate()
            
    def get(self,dataset):
        if dataset in self.datasets:
            return self.data[dataset].get()
        else:
            raise KeyError
            
    def onehot(self, string):
        onehot = np.zeros([len(string),vsize])
        onehot[np.arange(len(string)), np.array([index(char) for char in string])]=1
        return [onehot[i,:] for i in range(len(onehot))]
            
store = reader("/home/moon/data/",start=False)


# ## helper functions

# In[3]:


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


# In[4]:


#cell = td.ScopedLayer(tf.contrib.rnn.BasicLSTMCell(num_units=16), 'char_cell')


# In[5]:


def bidirectional_conv_LSTM():
    convlstm = Conv1DLSTMCell(input_shape=[vsize,1], output_channels=8, kernel_shape=[5])
    conv_lstm_cell_1d = td.ScopedLayer(convlstm)

    bidir_conv_lstm = td.Composition()
    with bidir_conv_lstm.scope():

        forward = td.Identity().reads(bidir_conv_lstm.input[0])
        backward = td.Identity().reads(bidir_conv_lstm.input[1])

        forw = (td.RNN(conv_lstm_cell_1d) >>
                td.GetItem(1) >>
                td.GetItem(0) >>
                td.Function(lambda rnn_outs: tf.contrib.layers.flatten(rnn_outs))).reads(forward)

        backw = (td.RNN(conv_lstm_cell_1d) >>
                 td.GetItem(1) >>
                 td.GetItem(0) >>
                 td.Function(lambda rnn_outs: tf.contrib.layers.flatten(rnn_outs))).reads(backward)

        rnn_outs = td.Concat().reads(forw,backw)
        bidir_conv_lstm.output.reads(rnn_outs)
    return bidir_conv_lstm >> td.FC(1)


def FCNN():
    return td.FC(50) >> td.FC(1)# >> td.Function(lambda xs: tf.squeeze(xs, axis=1))

data = td.Record((td.Map(
                        td.Vector(vsize) >>
                        td.Function(lambda x: tf.reshape(x, [-1,vsize,1]))),
                  td.Map(
                        td.Vector(vsize) >>
                        td.Function(lambda x: tf.reshape(x, [-1,vsize,1])))))
bidir =  data >> bidirectional_conv_LSTM()
#fc = FCNN()
blk = bidir# >> fc 


# In[6]:


compiler = td.Compiler.create((blk, td.Scalar()))
model_output, target = compiler.output_tensors
loss = tf.nn.l2_loss(model_output - target)
opt = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = opt.minimize(loss)
sess.run(tf.global_variables_initializer())



# In[ ]:


sess.run(tf.global_variables_initializer())


# In[ ]:

"""
losses = []
for i in range(1000):
    loff, _ = sess.run([loss, train_op], compiler.build_feed_dict([store.get("train") for _ in range(100)]))
    losses.append(loss)
    if i%10==0:
        print(loff)
"""

# In[ ]:
x = ([onehot("sadfasdfasd asdfas daf"), onehot("asdfasdf sdcfsd bdfovk")], 0.8)
fd = compiler.build_feed_dict([x for _ in range(1000)])
sess.run(tf.global_variables_initializer())
for i in range(10000):
    lof, _ = sess.run([loss, train_op], fd)
    print(lof)
#store.stop()
