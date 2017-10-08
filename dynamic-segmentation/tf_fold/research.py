
# coding: utf-8

# # Dynet segmentation with tf fold
# ![animation](../../fold/tensorflow_fold/g3doc/animation.gif)  

# In[2]:


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
data_dir = "/home/moon/data/"

#our alphabet

vocabulary = data.vocabulary(data_dir + 'vocabulary')
vsize=len(vocabulary)
print(vocabulary)

index = lambda char: vocabulary.index(char)
char = lambda i: vocabulary[i]


# In[3]:


def reader(queue, file):
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

qsize= 10000
pool = dict()
data = dict()
for dataset in ["train", "test", "validation"]:
    data[dataset] = Queue(qsize)
    pool[dataset] = Process(target=reader, args=(data[dataset], data_dir+dataset))
    pool[dataset].start()
    


# In[4]:


print(data["train"].get())

#p.terminate()


# ## helper functions

# In[5]:


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


# In[6]:


length = td.Length()

embedded =  (td.InputTransform(lambda s: [index(x) for x in s]) >> 
             td.Map(td.Scalar(tf.int32) >> 
             td.Function(td.Embedding(vsize, EMBEDDING_SIZE))))

onehot = (td.InputTransform(lambda s: [index(x) for x in s]) >>
          td.Map(td.Scalar(tf.int32) >>
          td.Function(lambda indices: tf.one_hot(indices, depth=vsize)) >>
          td.Function(lambda x: tf.reshape(x, [-1,vsize,1]))))

decode_onehot = td.InputTransform(lambda s: [char(np.argmax(np.squeeze(x))) for x in s])

print(decode_onehot.eval(onehot.eval("malú")))
#embedded.eval("kacsa")


# In[7]:


def conv1d_on_sequence(x, scope, kernel_size=3, input_channels=72, output_channels=72):
    with tf.variable_scope(scope) as sc:
        filters = tf.get_variable("conv_filter", [kernel_size] +  [input_channels, output_channels] , initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("conv_bias",  output_channels, initializer=tf.constant_initializer(0.05, dtype=tf.float32))
        conv = tf.nn.conv1d(x, filters=filters, stride=1, padding='VALID')
        return tf.nn.relu(tf.add(conv, bias))
    
def SeqToTuple(T, N):
    return (td.InputTransform(lambda x: tuple(x))
            .set_input_type(td.SequenceType(T))
            .set_output_type(td.Tuple(*([T] * N))))   


# In[8]:


convlstm = Conv1DLSTMCell(input_shape=[vsize,1], output_channels=8, kernel_shape=[5])
conv_lstm_cell_1d = td.ScopedLayer(convlstm)


# In[9]:


comp = td.Composition()
with comp.scope():
    #forward = td.Identity().reads(comp.input[0])
    #backward = td.Identity().reads(comp.input[1])
    model1 = (td.InputTransform(lambda s: [index(x) for x in s]) >>
             td.Map(td.Scalar(tf.int32) >>
             td.Function(lambda indices: tf.one_hot(indices, depth=vsize)) >>
             td.Function(lambda x: tf.reshape(x, [-1,vsize,1]))) >>             #onehot encoding
             td.RNN(conv_lstm_cell_1d)  >> td.GetItem(1) >> td.GetItem(0) ).reads(comp.input)
    model2 = (td.InputTransform(lambda s: [index(x) for x in s]) >>
             td.Map(td.Scalar(tf.int32) >>
             td.Function(lambda indices: tf.one_hot(indices, depth=vsize)) >>
             td.Function(lambda x: tf.reshape(x, [-1,vsize,1]))) >>             #onehot encoding
             td.RNN(conv_lstm_cell_1d) >> td.GetItem(1) >> td.GetItem(0) ).reads(comp.input)
    
    rnn_outs = td.Concat().reads(model1, model2)
    out = td.Function(lambda rnn_outs: tf.contrib.layers.flatten(rnn_outs)).reads(rnn_outs)
    #fc = out >> td.FC(300) >> td.FC(1)
    comp.output.reads(out)
    
new = comp >> td.FC(300) >>td.FC(200) >> td.FC(30) >> td.FC(1, activation=tf.nn.sigmoid)


# In[14]:


#new.eval(("cucc", "valami")).shape
compiler = td.Compiler.create(new)
(model_output,) = compiler.output_tensors
loss = tf.nn.l2_loss(model_output)
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)


# In[13]:


#print(new.eval("ss"))
print(new.input_type, new.output_type)
#sess.run(tf.global_variables_initializer())
#compiler.loom_input_tensor
#sess.run(tf.local_variables_initializer())
#sess.run(train_op,{compiler.loom_input_tensor: ["aa","bb"]})


# In[ ]:


fd = compiler.build_feed_dict(["nyam","mam", "ouf"])
sess.run(tf.global_variables_initializer())
sess.run(train_op, feed_dict=fd)

