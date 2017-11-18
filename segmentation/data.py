import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags


class Data:
    def __init__(self, folder, truncate):
        self.data_dir = folder
        self.data = dict()
        self.size = dict()
        self.datasets = ["train", "test", "validation"]
        self.truncate = truncate

        self.vocabulary = Data.vocabulary(self.data_dir + 'vocabulary')
        self.vsize = len(self.vocabulary)
        print(self.vocabulary)

        for dataset in self.datasets:
            self.data[dataset] = self.sentence_reader(folder + dataset)
            self.size[dataset] = sum(1 for line in open(folder + dataset))

    def index(self, char):
        return self.vocabulary.index(char)

    def char(self, index):
        return self.vocabulary[index]

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
                        sent_onehot = sent_onehot[:self.truncate]
                        tags = tags[:self.truncate]
                    yield (sent_onehot, tags)

    def onehot(self, string):
        onehot = np.zeros([len(string), self.vsize])
        indices = np.arange(len(string)), np.array([int(self.index(char)) for char in string])
        onehot[indices] = 1
        return [onehot[i, :] for i in range(len(onehot))]

    @staticmethod
    def pad(record):
        pads = ((FLAGS.truncate - len(record[1]), 0), (0, 0))
        ins = np.pad(record[0], pad_width=pads, mode="constant", constant_values=0)
        outs = np.pad(record[1], pad_width=(FLAGS.truncate - len(record[1]), 0), mode="constant", constant_values=0)
        return ins, outs

    def get_padded_batch(self, dataset="train"):
        data = np.zeros([FLAGS.batch_size, FLAGS.truncate, self.vsize])
        labels = np.zeros([FLAGS.batch_size, FLAGS.truncate, 1])
        for i in range(FLAGS.batch_size):
            sentence, label = Data.pad(next(self.data[dataset]))
            data[i] = sentence
            labels[i, :, 0] = label
        return data, labels

    @staticmethod
    def vocabulary(path):
        with open(path) as f:
            return sorted(set([char for char in f.readline().replace("\n", "")]))

