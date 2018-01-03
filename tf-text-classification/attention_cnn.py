import tensorflow as tf
import numpy as np

"""
One layer CNN with attention.

"""

DEBUG = False

class AttentionCNN(object):
    def __init__(self, document_length=120, embedding_dimension=400, n_classes=3, embedd_filter_sizes=[2, 3, 4],
                n_embedd_filters=100, n_dense_output=200, attention_depth=50):
        """
        One layer CNN with attention.

        :param document_length: Number of tokens per sentence (default: 120)
        :param embedding_dimension: Dimension of word embeddings (default: 400)
        :param n_classes: Number of classes/output nodes (default: 3)
        :param embedd_filter_sizes: Size of convolutional filters over document matrix (default: [2, 3, 4])
        :param n_embedd_filters: Number of convolutional filters (default: 100)
        :param n_dense_output: Number of nodes in dense layer (default: 100)
        :param attention_depth: Depth of attention vector (default: 50)
        """

        doc_embedding_size = embedding_dimension * document_length

        # place holders for input and output
        self.x = tf.placeholder(tf.float32, [None, doc_embedding_size], name="input_x")
        self.y_ = tf.placeholder(tf.float32, [None, n_classes], name="input_y_")
        self.keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")

        print("Weight Init: Xavier")

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
        self.doc_matrix = tf.reshape(x1, [-1,document_length,embedding_dimension])
        self.doc_matrix_expanded = tf.expand_dims(self.doc_matrix, -1)

        with tf.name_scope("attention"):
            U_shape = [embedding_dimension, attention_depth]  # (400(embedd_dim), 50(att_depth))
            #self.U_att = tf.Variable(tf.truncated_normal(U_shape, stddev=0.1), name="U_att")
            self.U_att = tf.get_variable("U_att", U_shape, initializer=tf.contrib.layers.xavier_initializer())

            def fn_matmul_w2v(previous_output, current_input):
                # print(current_input.get_shape())
                current_ouput = tf.matmul(current_input, self.U_att)
                # print 'previous_output', previous_output
                # print 'current_ouput', current_ouput
                return current_ouput

            #Creat w2v attention matrix
            initializer = tf.constant(np.zeros([document_length, attention_depth]), dtype=tf.float32)
            WU_att = tf.scan(fn_matmul_w2v, self.doc_matrix, initializer=initializer)
            if DEBUG:
                print('[WU_att]: ' + str(WU_att))   # (120(seq_len), 50(depth))

            WU_att_expanded = tf.expand_dims(WU_att, -1)
            if DEBUG:
                print('[WU_att_expanded]: ' + str(WU_att_expanded))  # (?, 120(seq_len), 50(att_depth), 1)

            #Pool w2v attention vector
            w2v_pool = tf.nn.max_pool(
                WU_att_expanded,
                ksize=[1, 1, attention_depth, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="w2v_pool")
            if DEBUG:
                print('[w2v_pool]: ' + str(w2v_pool))  # (?, 120(seq_len), 1, 1) #select attention for w2v

            self.att_pool_sq = tf.expand_dims(tf.squeeze(w2v_pool, squeeze_dims=[2, 3]), -1, name="att_pool_sq")
            if DEBUG:
                print('[att_pool_sq]: ' + str(self.att_pool_sq))  # (?, 120(seq_len), 1)

            #Multiply into w2v embedding attention vector
            self.doc_matrix_tr = tf.matrix_transpose(self.doc_matrix)
            if DEBUG:
                print('[doc_matrix_tr]: ' + str(self.doc_matrix_tr)) # (?, 400(embedd_dim), 120(seq_len))
            self.embedd_attention = tf.matmul(self.doc_matrix_tr, self.att_pool_sq, name="embedd_att")

            self.embedd_attention_sq = tf.squeeze(self.embedd_attention, squeeze_dims=[2], name="embedd_att_sq")
            if DEBUG:
                print('[attention_vector_sq]: ' + str(self.embedd_attention_sq)) # (?, 400)

        #sys.exit()
        conv_outputs = []

        # CNN layer for word embeddings
        # -------------------
        filter_index = 0
        for filter_size in embedd_filter_sizes:     
            # conv layer 10
            # first convolution over word embeddings
                               
            W1_conv = weight_variable("W1_conv_"+str(filter_index), [filter_size, embedding_dimension, 1, n_embedd_filters])
            b1_conv = bias_variable([n_embedd_filters])
            
            h1_conv = tf.nn.relu(conv2d(self.doc_matrix_expanded, W1_conv) + b1_conv)
            h1_pool = max_pool_2x2(h1_conv, document_length - filter_size + 1, 1)
            
            
            h1_norm = tf.nn.local_response_normalization(h1_pool)
            
            h1_norm_flat = tf.reshape(h1_norm, [-1, 1 * 1 * n_embedd_filters])
            
            conv_outputs.append(h1_norm_flat)
            filter_index = filter_index + 1

            
        # Append attention to pooling outputs
        conv_outputs.append(self.embedd_attention_sq)

        m_fc = tf.concat(axis=1, values=conv_outputs)

        # dropout
        m_fc_drop = tf.nn.dropout(m_fc, self.keep_prob)

        # dense layer (input is pooling outputs and embedding attention vector)
        new_size = len(embedd_filter_sizes)* n_embedd_filters + embedding_dimension

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


class AttentionCNNMapping(object):
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