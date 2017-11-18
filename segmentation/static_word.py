import numpy as np
import time
from data import Data
import tqdm
import tensorflow as tf
import segmentation.model as model
import segmentation.tools as tools

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size',    256, """batchsize""")
tf.app.flags.DEFINE_integer('epochs',        10, """epoch count""")
tf.app.flags.DEFINE_integer('truncate',      30, """truncate input sequences to this length""")
tf.app.flags.DEFINE_string('data_dir',       "/mnt/permanent/Home/nessie/velkey/data/", """data store basedir""")
tf.app.flags.DEFINE_string('log_dir',        "/mnt/permanent/Home/nessie/velkey/logs/", """logging directory root""")
tf.app.flags.DEFINE_string('run_name',       "new_word_3x100_static_conv_20_1_trun30", """naming: loss_fn, batch size, architecture, optimizer""")
tf.app.flags.DEFINE_string('model',          "lstm", """can be lstm, convlstm""")

tf.app.flags.DEFINE_string('loss',           "crossentropy", """can be l1, l2, crossentropy""")
tf.app.flags.DEFINE_string('optimizer',      "ADAM", """can be ADAM, RMS, SGD""")
tf.app.flags.DEFINE_float('learning_rate',   0.005, """starting learning rate""")

store = Data(FLAGS.data_dir + "word/", FLAGS.truncate)

x = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.truncate, store.vsize))
y = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, FLAGS.truncate, 1))
labels = y

if FLAGS.model == "lstm":
    rnn = model.stacked_fc_bi_lstm(x, [128, 128, 128])
elif FLAGS.model == "convlstm":
    rnn = model.stacked_fully_conv_bi_lstm(x, 20, [128, 128, 128],store.vsize)
    
logits = model.convolutional_output(rnn, [100,20,1], [5,5,3])
predictions = tf.nn.sigmoid(logits)


valid_chars_in_batch = tf.reduce_sum(x)
all_chars_in_batch = tf.size(x) / store.vsize
valid_ratio = valid_chars_in_batch / tf.cast(all_chars_in_batch, tf.float32)


loss = tools.loss(logits, labels)


path = FLAGS.log_dir + FLAGS.run_name
writer = tf.summary.FileWriter(path, graph=tf.get_default_graph())
saver = tf.train.Saver(max_to_keep=20)
tf.summary.scalar("batch_loss", loss)

precision, recall, accuracy, f1, predicted = tools.metrics(predictions, labels, x)
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
    good_words = 0
    step = int(store.size[dataset] / FLAGS.batch_size)
    for i in tqdm.trange(step):
        x_, y_ = store.get_padded_batch(dataset)
        feed = {
            x: x_,
            y: y_}
        batch_loss, acc, rec, f, preds = sess.run([loss, accuracy, recall, f1, predicted], feed_dict=feed)
        losses.append(batch_loss)
        accs.append(acc)
        recalls.append(rec)
        f1s.append(f)
        good_words += sum([np.all(y_[i, :] == preds[i, :]) for i in range(FLAGS.batch_size)])

    l, a, r, f = np.average(losses), np.average(accs), np.average(recalls), np.average(f1s)
    word_acc = good_words / store.size[dataset] * 100

    if dataset == "validation":
        feed = {validation_loss_placeholder: l,
                validation_accuracy_placeholder: float(a),
                validation_f1_placeholder: f}
        vl, va, vf = sess.run([validation_loss_summary, validation_accuracy_summary, validation_f1_summary],
                              feed_dict=feed)
        writer.add_summary(vl, train_step)
        writer.add_summary(va, train_step)
        writer.add_summary(vf, train_step)
    elif dataset == "test":
        feed = {test_loss_placeholder: l,
                test_accuracy_placeholder: float(a),
                test_f1_placeholder: f}
        vl, va, vf = sess.run([test_loss_summary, test_accuracy_summary, test_f1_summary], feed_dict=feed)
        writer.add_summary(vl, train_step)
        writer.add_summary(va, train_step)
        writer.add_summary(vf, train_step)
        writer.flush()

    return l, a, r, f, word_acc

    
early = tools.Stopper(40)
steps = FLAGS.epochs * int(store.size["train"] / FLAGS.batch_size)

# run training

sess.run(tf.global_variables_initializer())
for i in tqdm.trange(steps, unit="batches"):
    b_data, b_label = store.get_padded_batch("train")
    _, batch_loss, summary = sess.run([train_op, loss, summary_op], {x: b_data.astype(float), y: b_label})
    assert not np.isnan(batch_loss)
    
    if i % 20 == 0:
        writer.add_summary(summary, i)
        
    if i % int(steps / FLAGS.epochs / 2) == 0:
        l, a, r, f, w = get_metrics_on_dataset("validation", i)
        print("loss: ", l, " accuracy: ", a, "% recall: ", r, "fscore", f, " word_acc: ", w, "%")
        if early.add(l):
            break
            
    if i % int(steps / FLAGS.epochs / 2) == 0:
        save_path = saver.save(sess, path + "/model.ckpt", global_step=i)
        
print("Testing...")
l, a, r, f, w = get_metrics_on_dataset("test", steps)
print("loss: ", l, " accuracy: ", a, "% recall: ", r, "fscore", f, " word_acc: ", w, "%")

writer.flush()
