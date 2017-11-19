import numpy as np
import time
from data import Data
import tqdm
import tensorflow as tf
import random
import segmentation.model as model
import segmentation.tools as tools

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 256, """batchsize""")
tf.app.flags.DEFINE_integer('epochs', 30, """epoch count""")
tf.app.flags.DEFINE_integer('workers', 1, """epoch count""")
tf.app.flags.DEFINE_integer('truncate', 300, """truncate input sequences to this length""")
tf.app.flags.DEFINE_string('data_dir', "/mnt/permanent/Home/nessie/velkey/data/", """data store basedir""")
tf.app.flags.DEFINE_string('log_dir', "/mnt/permanent/Home/nessie/velkey/logs/sentence_conv/", """logging directory root""")
tf.app.flags.DEFINE_string('run_name', "", """""")
tf.app.flags.DEFINE_string('model', "convlstm", """can be lstm, convlstm""")

tf.app.flags.DEFINE_string('loss', "crossentropy", """can be l1, l2, crossentropy""")
tf.app.flags.DEFINE_string('optimizer', "ADAM", """can be ADAM, RMS, SGD""")
tf.app.flags.DEFINE_float('learning_rate', 0.005, """starting learning rate""")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def train_convlstm_model(kernel_size, unit_list, kernel_sizes, output_channels):
    tf.reset_default_graph()
    sess = tf.Session(config=config)
    re = lambda x: x.replace(" ", "").replace("[", "_").replace("]", "_").replace(",", "_")
    NAME = "sentence_convlstm_" + re(str(unit_list)) + "_conv_channel_" + re(str(output_channels)) + "_kernels_" + re(
        str(kernel_sizes))

    store = Data(FLAGS.data_dir + "sentence/", FLAGS.truncate)

    x = tf.placeholder(tf.float32, shape=(None, FLAGS.truncate, store.vsize), name="input_x")
    y = tf.placeholder(tf.int32, shape=(None, FLAGS.truncate, 1), name="input_y")
    labels = y

    rnn = model.stacked_fully_conv_bi_lstm(x, kernel_size, unit_list, store.vsize)

    logits = model.convolutional_output(rnn, output_channels, kernel_sizes)
    predictions = tf.nn.sigmoid(logits, name='output_probs')

    valid_chars_in_batch = tf.reduce_sum(x)
    all_chars_in_batch = tf.size(x) / store.vsize
    valid_ratio = valid_chars_in_batch / tf.cast(all_chars_in_batch, tf.float32)

    loss = tools.loss(logits, labels)

    path = FLAGS.log_dir + NAME
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
        good_sents = 0
        words = 0
        good_words = 0
        step = int(store.size[dataset] / FLAGS.batch_size)
        for i in tqdm.trange(step):
            x_, y_ = store.get_padded_batch(dataset)
            feed = {
                x: x_,
                y: y_}

            batch_loss, rec, f, preds = sess.run([loss, recall, f1, predicted], feed_dict=feed)
            losses.append(batch_loss)
            accs.append(tools.char_accuracy_on_padded(x_, y_, preds, store.vsize))
            recalls.append(rec)
            f1s.append(f)
            good_sents += sum([np.all(y_[i, :] == preds[i, :]) for i in range(FLAGS.batch_size)])


            for i in range(FLAGS.batch_size):
                X = x_[i]
                Y = np.squeeze(y_[i])
                P = np.squeeze(preds[i])
                wl = []
                wp = []
                le = int(np.sum(X))
                
                for char, lab, prd in zip(X[-le:], Y[-le:], P[-le:]):
                    if np.where(char == True)[0][0] != store.index(" "):
                        wl.append(lab)
                        wp.append(prd)
                    elif len(wl) != 0:
                        words += 1
                        good_words += (wl == wp)                       
                        wl = []
                        wp = []
                        
        if words != 0:
            word_acc = good_words / words * 100
        else: 
            word_acc = 0
            
        l, a, r, f = np.average(losses), np.average(accs), np.average(recalls), np.average(f1s)
        sent_acc = good_sents / store.size[dataset] * 100

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

        return l, a, r, f, word_acc ,sent_acc

    early = tools.Stopper(30)
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
            l, a, r, f, w, s = get_metrics_on_dataset("validation", i)
            print("loss: ", l, " accuracy: ", a, "% recall: ", r, "fscore", f, " word_acc: ", w, "%", " sent_acc: ", s, "%")
            if early.add(l):
                break

        if i % int(steps / FLAGS.epochs / 2) == 0:
            save_path = saver.save(sess, path + "/model.ckpt", global_step=i)

    print("Testing...")
    l, a, r, f, w, s = get_metrics_on_dataset("test", steps)
    print("loss: ", l, " accuracy: ", a, "% recall: ", r, "fscore", f, " word_acc: ", w, "%", " sent_acc: ", s, "%")
    # log test losses:
    with open(FLAGS.log_dir + "hyper.log", "a") as myfile:
        myfile.write("\n" + FLAGS.log_dir + NAME + "\t" + str(l) + "\t" + str(a) + "\t" + str(r) + "\t" + str(f) + "\t" + str(w) + "\t" + str(s))
    writer.flush()
    sess.close()

    return l, a, r, f, w


def sample_hyper():
    recurrent_kernel = [20, 30, 60, 90]
    kernel_opt = [1, 3, 5, 7, 9]
    ch_opt = [4, 8, 16, 32, 64]
    unit_opt = [64, 128, 256]

    lstm_depth_range = (1, 2)
    conv_depth_range = (1, 2)
    
    kern = random.choice(recurrent_kernel)

    num_units = [random.choice(unit_opt)]
    for lstm_cell in range(random.randint(*lstm_depth_range)):
        num_units.append(random.choice(unit_opt[:unit_opt.index(min(num_units))+1]))

    kernels = [random.choice(kernel_opt)]
    channels = [random.choice(ch_opt)]
    for kernel in range(random.randint(*conv_depth_range)):
        kernels.append(random.choice(kernel_opt))
        channels.append(random.choice(ch_opt[:ch_opt.index(min(channels))+1]))

    channels[-1] = 1

    return kernels, channels, num_units, kern


def train(settings):
    return train_convlstm_model(kernel_size=settings[3], unit_list=settings[2], kernel_sizes=settings[0], output_channels=settings[1])

def hyperopt(workers=2):
    from multiprocessing import Pool

    def gen(count):
        for i in range(count):
            yield sample_hyper()

    setting = gen(2000)

    po = Pool(workers)
    po.map(train, setting)


def main(argv=None):
    hyperopt(FLAGS.workers)
    # train_convlstm_model(unit_list=[128,128,128], kernel_sizes=[5,3,1], output_channels=[100,20,1])


if __name__ == "__main__":
    tf.app.run()

