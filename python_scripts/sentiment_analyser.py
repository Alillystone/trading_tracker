import tensorflow as tf
import numpy as np
import pandas as pd
import random
import pickle
from collections import Counter
from random import randint
from PyDictionary import PyDictionary


class SentimentAnalyser:
    def __init__(self, 
                 min_neurons, 
                 max_neurons, 
                 num_hidden_layers):
        self.min_n_nodes    = min_neurons
        self.max_n_nodes    = max_neurons
        self.n_node_layers  = num_hidden_layers
        self.n_output_nodes = 2
        self.n_epochs       = 10
        self.batch_size     = 120

    def initialise_word_map(self, file_list):

        lower_count_limit = 30
        upper_count_limit = 1000

        word_map = []
        for file in file_list:
            contents = None
            with open(file,'r') as textfile:
                contents = textfile.readlines()

            for line in contents:
                word_map += list(line.split())

        word_counts  = Counter(word_map)
        lim_word_map = []
        for word in word_counts:
            if upper_count_limit > word_counts[word] > lower_count_limit:
                lim_word_map.append(word)
        self.word_map = lim_word_map

        self.word_type_map = ['Noun', 'Verb', 'Adverb', 'Preposition',
                              'Pronoun', 'Adjective', 'Conjunction',
                              'Determiner', 'Exclamation', 'Punctuation',
                              'Unknown']
        self.meaning_array = np.zeros(shape=(len(self.word_map),
                                             len(self.word_type_map)))
        dictionary = PyDictionary()
        for word_index, word in enumerate(self.word_map):
            word_meaning = dictionary.meaning(word)
            if (not word_meaning and len(word) == 1):
                word_meaning = {'Punctuation' : ['unknown']}
            if (not word_meaning and len(word) > 1):
                word_meaning = {'Unknown' : ['unknown']}
            for word_type, description in word_meaning.items():
                type_index = self.word_type_map.index(word_type)
                self.meaning_array[word_index][type_index] += len(description)

    def prepare_training_data(self):

        files = {'DIR' : ['sentiment_data/positive/pos.txt',
                          'sentiment_data/negative/neg.txt'],
                 'CLF' : [[1,0],
                          [0,1]]}

        self.initialise_word_map(files['DIR'])

        def handle_sample(sample, classification):

            contents = None
            with open(sample,'r') as textfile:
                contents = textfile.readlines()

            feature_shape = (len(self.word_map), len(self.word_type_map) + 1)
            feature_set = []
            for line in contents:
                features = np.zeros(shape=feature_shape)
                for word in line.split():
                    if word not in self.word_map:
                        continue
                    index_value = self.word_map.index(word)
                    features[index_value][0]  += 1
                    features[index_value][1:] += self.meaning_array[index_value]
                features = list(features.flatten())
                feature_set.append([features, classification])
            return feature_set

        feature_set = []
        for idx in range(len(files)):
            feature_set += handle_sample(files['DIR'][idx], files['CLF'][idx])
        random.shuffle(feature_set)

        self.test_size = 0.1
        feature_set = np.array(feature_set)
        testing_size = int(self.test_size * len(feature_set))

        self.train_x = list(feature_set[:,0][:-testing_size])
        self.train_y = list(feature_set[:,1][:-testing_size])
        self.tests_x = list(feature_set[:,0][-testing_size:])
        self.tests_y = list(feature_set[:,1][-testing_size:])

    def create_layer(self, n_input_nodes, prev_layer, output=False):

        if not output:
            n_output_nodes = randint(self.min_n_nodes, self.max_n_nodes)
        else:
            n_output_nodes = self.n_output_nodes

        layer_dict = {'weights':tf.Variable(
                                tf.random_normal([n_input_nodes,
                                                  n_output_nodes])),
                      'biases' :tf.Variable(
                                tf.random_normal([n_output_nodes]))}

        if not output:
            layer = tf.add(tf.matmul(prev_layer,
                                     layer_dict['weights']), 
                                     layer_dict['biases'])
            layer = tf.nn.relu(layer)
        else:
            layer = tf.matmul(prev_layer,
                              layer_dict['weights']) + layer_dict['biases']

        return n_output_nodes, layer

    def build_neural_model(self):
        self.n_input_nodes = len(self.train_x[0])
        self.x = tf.placeholder('float')
        self.y = tf.placeholder('float')

        prev_n_nodes, prev_layer = self.create_layer(self.n_input_nodes,
                                                     self.x)

        for n_layers in range(0, self.n_node_layers):
            prev_n_nodes, prev_layer = self.create_layer(prev_n_nodes,
                                                         prev_layer)

        prev_n_nodes, output = self.create_layer(prev_n_nodes,
                                                 prev_layer,
                                                 output=True)

        return output

    def train_neural_network(self):
        prediction = self.build_neural_model()
        cost = tf.reduce_mean(
               tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, 
                                                          labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.n_epochs):
                epoch_loss = 0
                i = 0
                while (i < len(self.train_x)):
                    start = i
                    end   = i + self.batch_size

                    batch_x = np.array(self.train_x[start:end])
                    batch_y = np.array(self.train_y[start:end])

                    _, c = sess.run([optimizer, cost], 
                                     feed_dict={self.x: batch_x, 
                                                self.y: batch_y})
                    epoch_loss += c

                    i += self.batch_size

                print('Epoch', epoch + 1, 
                      'completed out of', self.n_epochs,
                      'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',accuracy.eval({self.x:self.tests_x,
                                             self.y:self.tests_y}))