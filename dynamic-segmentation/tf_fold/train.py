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
tf.app.flags.DEFINE_integer('batch_size',    8, """batchsize""")
tf.app.flags.DEFINE_integer('epochs',        10, """epoch count""")
tf.app.flags.DEFINE_string('data_dir',       "/mnt/permanent/Home/nessie/velkey/data/", """data store basedir""")
tf.app.flags.DEFINE_string('log_dir',        "/mnt/permanent/Home/nessie/velkey/logs/", """logging directory root""")
tf.app.flags.DEFINE_string('run_name',       "development", """naming: loss_fn, batch size, architecture, optimizer""")
tf.app.flags.DEFINE_string('data_type',      "sentence/", """can be sentence/, word/""")
tf.app.flags.DEFINE_string('model',          "lstm", """can be lstm, convlstm right now""")
tf.app.flags.DEFINE_string('loss',           "crossentropy", """can be l1, l2, crossentropy""")
tf.app.flags.DEFINE_string('optimizer',      "ADAM", """can be ADAM, RMS, SGD""")
tf.app.flags.DEFINE_float('learning_rate',   0.001, """starting learning rate""")


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

def model_information():
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
    return td.ScopedLayer(tf.contrib.rnn.MultiRNNCell(
        [convLSTM_cell(kernel, features)
         for (kernel, features) in zip(kernel_sizes, out_features)]))

def FC_cell(units):
    return td.ScopedLayer(tf.contrib.rnn.LSTMCell(num_units=units))

def multi_FC_cell(units_list):
    return td.ScopedLayer(tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_units=units) for units in units_list]))
    
def bidirectional_dynamic_CONV(fw_cell, bw_cell, out_features=64):
    bidir_conv_lstm = td.Composition()
    with bidir_conv_lstm.scope():        
        fw_seq = td.Identity().reads(bidir_conv_lstm.input[0])
        labels = (td.GetItem(1)>>td.Map(td.Metric("labels"))>>td.Void()).reads(bidir_conv_lstm.input)
        bw_seq = td.Slice(step=-1).reads(fw_seq)

        forward_dir = (td.RNN(fw_cell) >> td.GetItem(0)).reads(fw_seq)
        back_dir = (td.RNN(bw_cell) >> td.GetItem(0)).reads(bw_seq)
        back_to_leftright = td.Slice(step=-1).reads(back_dir)
        
        output_transform = (td.Function(lambda x: tf.reshape(x, [-1,vsize*out_features])) >>
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
        fw_seq = td.Identity().reads(bidir_conv_lstm.input[0])
        labels = (td.GetItem(1)>>td.Map(td.Metric("labels"))>>td.Void()).reads(bidir_conv_lstm.input)
        bw_seq = td.Slice(step=-1).reads(fw_seq)

        forward_dir = (td.RNN(fw_cell) >> td.GetItem(0)).reads(fw_seq)
        back_dir = (td.RNN(bw_cell) >> td.GetItem(0)).reads(bw_seq)
        back_to_leftright = td.Slice(step=-1).reads(back_dir)
        
        output_transform = td.FC(1, activation=None)
        
        bidir_common = (td.ZipWith(td.Concat() >> 
                                  output_transform >> td.Metric('logits'))).reads(forward_dir, back_to_leftright)
        
        bidir_conv_lstm.output.reads(bidir_common)
    return bidir_conv_lstm

CONV_data = td.Record((td.Map(td.Vector(vsize) >> td.Function(lambda x: tf.reshape(x, [-1,vsize,1]))),td.Map(td.Scalar())))
CONV_model =  (CONV_data >>
               bidirectional_dynamic_CONV(convLSTM_cell(vsize), convLSTM_cell(vsize)) >>
               td.Void())

FC_data = td.Record((td.Map(td.Vector(vsize)),td.Map(td.Scalar())))
FC_model = (FC_data >>
            bidirectional_dynamic_FC(multi_FC_cell([100]*8), multi_FC_cell([100]*8),100) >>
            td.Void())

store = data(FLAGS.data_dir + FLAGS.data_type)

if FLAGS.model == "lstm":
    model = FC_model
elif FLAGS.model == "convlstm":
    model = CONV_model
else:
    raise NotImplemented
    
compiler = td.Compiler.create(model)
logits = tf.squeeze(compiler.metric_tensors['logits'])
labels = compiler.metric_tensors['labels']
predictions = tf.nn.sigmoid(logits)

l1_loss = tf.reduce_mean(tf.abs(tf.subtract(labels,logits)))
l2_loss = tf.reduce_mean(tf.abs(tf.subtract(labels,logits)))
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

if FLAGS.loss == "l1":
    loss = l1_loss
elif FLAGS.loss == "l2":
    loss = l2_loss
elif FLAGS.loss == "crossentropy":
    loss = cross_entropy
else:
    raise NotImplemented

path = FLAGS.log_dir + FLAGS.run_name
writer = tf.summary.FileWriter(path, graph=tf.get_default_graph())
saver = tf.train.Saver(max_to_keep=20)
tf.summary.scalar("batch_loss", loss)

#Accuracy
acc = tf.reduce_sum(tf.cast(tf.equal(tf.less(0.5,predictions), tf.cast(labels, tf.bool)),tf.int32))*100/tf.size(labels)
tf.summary.scalar("batch_accuracy", acc)

# Recall
correct_trues = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.less(0.5, predictions), tf.cast(labels, tf.bool)), tf.cast(labels, tf.bool)), tf.int32))
all_trues = tf.reduce_sum(labels)
recall = tf.cast(correct_trues,tf.float32) / all_trues
tf.summary.scalar("recall", recall)
         
summary_op = tf.summary.merge_all()

if FLAGS.optimizer == "ADAM":
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
elif FLAGS.optimizer == "RMS":
    opt = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
elif FLAGS.optimizer == "SGD":
    opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
else:
    raise NotImplemented

train_op = opt.minimize(loss)
sess.run(tf.global_variables_initializer())

# validation summary:
validation_loss_placeholder = tf.placeholder(tf.float32, name="validation")
validation_loss_summary = tf.summary.scalar('validation_loss', validation_loss_placeholder)
test_loss_placeholder = tf.placeholder(tf.float32, name="test")
test_loss_summary = tf.summary.scalar('validation_loss', test_loss_placeholder)


def get_metrics_on_dataset(dataset, train_step):
    loss_sum = 0
    accs = []
    recalls = []
    for i in tqdm.trange(int(store.size[dataset] / FLAGS.batch_size)):
        batch_loss, accuracy, rec = sess.run([loss, acc, recall],
                              compiler.build_feed_dict([next(store.data[dataset]) for _ in range(FLAGS.batch_size)]))
        loss_sum += batch_loss
        accs.append(accuracy)
        recalls.append(rec)
    
    if dataset == "validation":
        valid_summary = sess.run(validation_loss_summary,feed_dict={validation_loss_placeholder: loss_sum})
        writer.add_summary(valid_summary, train_step + 1)
    elif dataset == "test":
        test_summary = session.run(test_loss_summary,feed_dict={test_loss_placeholder: loss_sum})
        writer.add_summary(test_summary, train_step + 1)

    return loss_sum, np.average(accs), np.average(recalls)
    
    
class stopper():
    def __init__(self, patience=20):
        self.log = []
        self.patience = patience
        self.should_stop = False
        
    def add(self, value)
        self.log.append(value)
        return self.check()
    
    def check(self):
        minimum = min(self.log)
        errors = sum([1 if i>minimum else 0 for i in self.log[self.log.index(minimum):]])
        if errors > self.patience:
            self.should_stop = True
        return self.should_stop
    
early = stopper(20)

for i in tqdm.trange(FLAGS.epochs * int(store.size["train"] / FLAGS.batch_size), unit="batches"):
    _, batch_loss, summary = sess.run([train_op, loss, summary_op],
                                      compiler.build_feed_dict([next(store.data["train"])
                                                                for _ in range(FLAGS.batch_size)]))
    assert not np.isnan(batch_loss)
    
    if i % 5 == 0:
        writer.add_summary(summary, i)
        
    if i % 500 == 0:
        l, a, r = get_metrics_on_dataset("validation", i)
        print("loss: ", l, " accuracy: ", a, "% recall: ", r)
        if early.add(l):
            break
            
    if i % 1000 == 0:
        save_path = saver.save(sess, path + "/model.ckpt", global_step=i)
        
print("Testing...")
l, a, r = get_metrics_on_dataset("test")
print("loss: ", l, " accuracy: ", a, "% recall: ", r)
    

#DONE remove duplicate words
#DONE switch to more stable loss implementation (tf cross entropy)
#DONE automated validation and test
#DONE Early stopping
#DONE auto save
#DONE word training
#DONE command line usage
#TODO metrics : word level accuracy, char level accuracy DONE, recall F-score
#TODO inference ipynotebook