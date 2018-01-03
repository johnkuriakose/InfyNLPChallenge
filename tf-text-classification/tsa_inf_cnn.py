import tensorflow as tf
import numpy as np

"""
CNN architecture described as foundation of:
Deshmane, Friedrichs -2017- TSA-INF at SemEval-2017 Task 4: An Ensemble of Deep Learning Architectures Including Lexicon Features for Twitter Sentiment Analysis

"""


class TSAINFCNN(object):
    def __init__(self, document_length, embedding_dimension, n_classes, embedd_filter_sizes, n_embedd_filters,
                 n_dense_output):
        """
        One layer CNN used in SemEval competition 2017.

        :param document_length: Number of tokens per sentence
        :param embedding_dimension: Dimension of word embeddings
        :param n_classes: Number of classes/output nodes
        :param embedd_filter_sizes: Size of convolutional filters over document matrix
        :param n_embedd_filters: Number of convolutional filters
        :param n_dense_output: Number of nodes in dense layer
        """

        doc_embedding_size = embedding_dimension * document_length
        #TODO: legacy parameters from original implementation, remove to clean up
        doc_channels = 1
        input_channels = 1
        
        # place holders for input and output
        self.x = tf.placeholder(tf.float32, [None, doc_embedding_size], name="input_x")
        self.y_ = tf.placeholder(tf.float32, [None, n_classes], name="input_y_")
        self.keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        #print("Weight Init: Xavier")

        def weight_variable(name, shape):
          #initial = tf.truncated_normal(shape, stddev=0.1)
          #initial = tf.zeros(shape)
          W = tf.get_variable(name, shape,
                   initializer = tf.contrib.layers.xavier_initializer())
          return W

        def bias_variable(shape):
          #initial = tf.constant(0.1, shape=shape)
          initial = tf.zeros(shape)
          return tf.Variable(initial)
          
        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


        def max_pool_2x2(x, kernel_width, kernel_height):
          return tf.nn.max_pool(x, ksize=[1, kernel_width, kernel_height, 1],
                                strides=[1, 1, 1, 1], padding='VALID')

        x1 = self.x[:,0:doc_embedding_size]
        x1_image = tf.reshape(x1, [-1,document_length,embedding_dimension,doc_channels])

        conv_outputs = []

        # CNN layer for word embeddings
        # -------------------
        filter_index = 0
        for filter_size in embedd_filter_sizes:     
            # conv layer 10
            # first convolution over word embeddings
                               
            W1_conv = weight_variable("W1_conv_"+str(filter_index), [filter_size, embedding_dimension, input_channels, n_embedd_filters])
            b1_conv = bias_variable([n_embedd_filters])
            
            h1_conv = tf.nn.relu(conv2d(x1_image, W1_conv) + b1_conv)
            h1_pool = max_pool_2x2(h1_conv, document_length - filter_size + 1, 1)
            
            
            h1_norm = tf.nn.local_response_normalization(h1_pool)
            
            h1_norm_flat = tf.reshape(h1_norm, [-1, 1 * 1 * n_embedd_filters])
            
            conv_outputs.append(h1_norm_flat)
            filter_index = filter_index + 1

            

        m_fc = tf.concat(axis=1, values=conv_outputs)


        # dropout

        m_fc_drop = tf.nn.dropout(m_fc, self.keep_prob)

        # dense layer 
        new_size = len(embedd_filter_sizes)* n_embedd_filters

        W_fc = weight_variable("W_fc", [new_size, n_dense_output])
        b_fc = bias_variable([n_dense_output])
             
        h_fc = tf.nn.relu(tf.matmul(m_fc_drop, W_fc) + b_fc)

        h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

        # output layer

        W_fc2 = weight_variable("W_fc2", [n_dense_output, n_classes])
        b_fc2 = bias_variable([n_classes])

        self.scores = tf.nn.xw_plus_b(h_fc_drop, W_fc2, b_fc2, name="scores")
        self.y_conv = tf.nn.softmax(self.scores, name="predictions")
        self.preds = tf.argmax(self.y_conv, 1, name="preds")
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), axis=[1]), name="loss")


class TsaInfCnnMapping(object):
    """
    Make model accessible from graph object by mapping member variables.

    This class maps all named variables from a graph object to class member variables. Class member variables can thus
    be accessed just like in the original model.

    Use this to gain access to model variables when loading a checkpoint.
    """
    def __init__(self, graph, variable_scope=""):
        """
        Init class member variables by mapping named variables from graph.

        Optionally input an additional variable scope, for example when loading cross validation or annealing models.

        :param graph: tensorflow.Graph
            Tensorflow graph object
        :param variable_scope: string
            Additional variable scope identifier
        """
        if variable_scope != "":
            variable_scope = variable_scope + "/"
        self.x = graph.get_operation_by_name(variable_scope + "input_x").outputs[0]
        self.y_ = graph.get_operation_by_name(variable_scope + "input_y_").outputs[0]
        self.keep_prob = graph.get_operation_by_name(variable_scope + "input_keep_prob").outputs[0]

        # Tensors we want to evaluate
        self.scores = graph.get_operation_by_name(variable_scope + "scores").outputs[0]
        self.y_conv = graph.get_operation_by_name(variable_scope + "predictions").outputs[0]
        self.preds = graph.get_operation_by_name(variable_scope + "preds").outputs[0]
        self.cross_entropy = graph.get_operation_by_name(variable_scope + "loss").outputs[0]

