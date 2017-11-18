import numpy as np
import time
import data
import tqdm
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
import tensorflow_fold as td
from conv_lstm_cell import Conv1DLSTMCell


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size',    256, """batchsize""")
tf.app.flags.DEFINE_integer('epochs',        100, """epoch count""")
tf.app.flags.DEFINE_integer('truncate',      200, """truncate input sequences to this length""")
tf.app.flags.DEFINE_string('data_dir',       "/mnt/permanent/Home/nessie/velkey/data/", """data store basedir""")
tf.app.flags.DEFINE_string('log_dir',        "/mnt/permanent/Home/nessie/velkey/logs/", """logging directory root""")
tf.app.flags.DEFINE_string('run_name',       "sentence_3x128_static_conv_128_64_1_trun200", """naming: loss_fn, batch size, architecture, optimizer""")
tf.app.flags.DEFINE_string('data_type',      "sentence/", """can be sentence/, word/""")
tf.app.flags.DEFINE_string('model',          "lstm", """can be lstm, convlstm""")

tf.app.flags.DEFINE_string('loss',           "crossentropy", """can be l1, l2, crossentropy""")
tf.app.flags.DEFINE_string('optimizer',      "ADAM", """can be ADAM, RMS, SGD""")
tf.app.flags.DEFINE_float('learning_rate',   0.005, """starting learning rate""")


vocabulary = data.vocabulary(FLAGS.data_dir + 'vocabulary')
vsize=len(vocabulary)
print(vocabulary)

index = lambda char: vocabulary.index(char)
char = lambda i: vocabulary[i]


class data():
    def __init__(self, folder, truncate):
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
        read sentences from the data format setence: sentence\tlabels\n
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
            variable_parametes *= dim.value
        print("\tparams: ", variable_parametes)
        total_parameters += variable_parametes
    print(total_parameters)
    return total_parameters


store = data(FLAGS.data_dir + FLAGS.data_type, FLAGS.truncate)
def pad(record):
    pads = ((FLAGS.truncate-len(record[1]), 0), (0, 0))
    ins = np.pad(record[0], pad_width=pads, mode="constant", constant_values=0)
    outs = np.pad(record[1], pad_width=(FLAGS.truncate-len(record[1]), 0), mode="constant", constant_values=0)
    return (ins, outs)

def get_padded_batch(dataset="train"):
    data = np.zeros([FLAGS.batch_size, FLAGS.truncate, vsize])
    labels = np.zeros([FLAGS.batch_size, FLAGS.truncate, 1])
    for i in range(FLAGS.batch_size):
        sentence, label = pad(next(store.data[dataset]))
        data[i] = sentence
        labels[i, :, 0] = label
    return data, labels


x = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.truncate, vsize))
y = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, FLAGS.truncate, 1))
labels = y

#lstm = Conv1DLSTMCell(input_shape=[vsize,1], output_channels=units, kernel_shape=[kernel_size])

if FLAGS.model == "lstm":
    num_units = [128, 128, 128]
    
    with tf.variable_scope("fw"):
        fw_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=units) for units in num_units])
    with tf.variable_scope("bw"):
        bw_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=units) for units in num_units])
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cells, cell_bw=bw_cells, inputs=x, dtype=tf.float32)
    
    RNN_out = tf.concat(outputs, -1)
    
    layer_0 = tf.layers.conv1d(RNN_out, 1, 5, activation=tf.nn.sigmoid, padding='same')
    #layer_1 = tf.layers.conv1d(layer_0, 64, 3, activation=tf.nn.sigmoid, padding='same')
    #layer_2 = tf.layers.conv1d(layer_1, 1, 1, activation=tf.nn.sigmoid, padding='same')
    logits = layer_0
    #logits = tf.nn.conv1d(RNN_out,filters=filters,stride=1,padding='SAME') + bias
    
elif FLAGS.model == "convlstm":
    kernel_size = 20
    num_units = [100, 100, 100]
    
    with tf.variable_scope("fw"):
        fw_cells = tf.contrib.rnn.MultiRNNCell([Conv1DLSTMCell(input_shape=[vsize,1], output_channels=units, kernel_shape=[kernel_size]) for units in num_units])
    with tf.variable_scope("bw"):
        bw_cells = tf.contrib.rnn.MultiRNNCell([Conv1DLSTMCell(input_shape=[vsize,1], output_channels=units, kernel_shape=[kernel_size]) for units in num_units])
    
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cells, cell_bw=bw_cells, inputs=x, dtype=tf.float32)
    outputs = tf.reshape(outputs,-1)

          
predictions = tf.nn.sigmoid(logits)


valid_chars_in_batch = tf.reduce_sum(x)
all_chars_in_batch = tf.size(x) / vsize
valid_ratio = valid_chars_in_batch / tf.cast(all_chars_in_batch, tf.float32)

l1_loss = tf.reduce_mean(tf.abs(tf.subtract(logits, tf.cast(labels, tf.float32))))
l2_loss = tf.reduce_mean(tf.square(tf.subtract(logits, tf.cast(labels, tf.float32))))
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.float32)))

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

def metrics(probs, labels, x):
    labels = tf.cast(labels, tf.int32)
    predicted = tf.cast(tf.less(0.5, probs),tf.int32)
    length = tf.reduce_sum(x)
    
    #crop the sequences:
    labels = labels
    
    TP = tf.count_nonzero(predicted * labels)
    TN = tf.count_nonzero((predicted - 1) * (labels - 1))
    FP = tf.count_nonzero(predicted * (labels - 1))
    FN = tf.count_nonzero((predicted - 1) * labels)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
                          
    accuracy = tf.cast(tf.count_nonzero(tf.equal(predicted, labels)), tf.float32) / tf.cast(tf.size(labels), tf.float32) * 100
         
    return precision, recall, accuracy, f1
    
precision, recall, accuracy, f1 = metrics(predictions, labels,x)
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("precision", precision)
tf.summary.scalar("recall", recall)
tf.summary.scalar("f1", f1)
         
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
tf.global_variables_initializer()

validation_loss_placeholder = tf.placeholder(tf.float32, name="validation")
validation_loss_summary = tf.summary.scalar('validation_loss', validation_loss_placeholder)
validation_accuracy_placeholder = tf.placeholder(tf.float32, name="validation_accuracy")
validation_accuracy_summary = tf.summary.scalar('validation_accuracy', validation_accuracy_placeholder)
validation_f1_placeholder = tf.placeholder(tf.float32, name="validation_f1")
validation_f1_summary = tf.summary.scalar('validation_f1', validation_f1_placeholder)
test_loss_placeholder = tf.placeholder(tf.float32, name="test")
test_loss_summary = tf.summary.scalar('test_loss', test_loss_placeholder)
test_f1_placeholder = tf.placeholder(tf.float32, name="test_f1")
test_f1_summary = tf.summary.scalar('test_f1', test_f1_placeholder)
test_accuracy_placeholder = tf.placeholder(tf.float32, name="test_accuracy")
test_accuracy_summary = tf.summary.scalar('test_accuracy', test_accuracy_placeholder)

def get_metrics_on_dataset(dataset, train_step):
    losses = []
    accs = []
    recalls = []
    f1s = []
    step = int(store.size[dataset] / FLAGS.batch_size)
    for i in tqdm.trange(step):
        x_, y_ = get_padded_batch(dataset)
        feed = {
            x: x_,
            y: y_}
        batch_loss, acc, rec, f = sess.run([loss, accuracy, recall, f1],feed_dict=feed)
        losses.append(batch_loss)
        accs.append(acc)
        recalls.append(rec)
        f1s.append(f)
    
    l, a, r, f = np.average(losses), np.average(accs), np.average(recalls), np.average(f1s)
    
    if dataset == "validation":
        feed = {validation_loss_placeholder: l,
                validation_accuracy_placeholder: float(a),
                validation_f1_placeholder: f}
        vl, va, vf = sess.run([validation_loss_summary, validation_accuracy_summary, validation_f1_summary],feed_dict=feed)
        writer.add_summary(vl, train_step)
        writer.add_summary(va, train_step)
        writer.add_summary(vf, train_step)
    elif dataset == "test":
        feed = {test_loss_placeholder: l,
                test_accuracy_placeholder: float(a),
                test_f1_placeholder: f}
        vl, va, vf = sess.run([test_loss_summary, test_accuracy_summary, test_f1_summary],feed_dict=feed)
        writer.add_summary(vl, train_step)
        writer.add_summary(va, train_step)
        writer.add_summary(vf, train_step)
        writer.flush()

    return l,a,r,f
    
    
class stopper():
    def __init__(self, patience=20):
        self.log = []
        self.patience = patience
        self.should_stop = False
        
    def add(self, value):
        self.log.append(value)
        return self.check()
    
    def check(self):
        minimum = min(self.log)
        errors = sum([1 if i>minimum else 0 for i in self.log[self.log.index(minimum):]])
        if errors > self.patience:
            self.should_stop = True
        return self.should_stop
    
early = stopper(40)
steps = FLAGS.epochs * int(store.size["train"] / FLAGS.batch_size)

# run training

sess.run(tf.global_variables_initializer())
for i in tqdm.trange(steps, unit="batches"):
    b_data, b_label = get_padded_batch("train")
    _, batch_loss, summary= sess.run([train_op, loss, summary_op], {x: b_data.astype(float), y: b_label})
    assert not np.isnan(batch_loss)
    
    if i % 20 == 0:
        writer.add_summary(summary, i)
        
    if i % int(steps / FLAGS.epochs / 2) == 0:
        l, a, r, f = get_metrics_on_dataset("validation", i)
        print("loss: ", l, " accuracy: ", a, "% recall: ", r, "fscore: ", f)
        if early.add(l):
            break
            
    if i % int(steps / FLAGS.epochs / 2) == 0:
        save_path = saver.save(sess, path + "/model.ckpt", global_step=i)
        
print("Testing...")
l, a, r, f = get_metrics_on_dataset("test", steps)
print("loss: ", l, " accuracy: ", a, "% recall: ", r, "fscore", f)

writer.flush()
