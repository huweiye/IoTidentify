from base import BaseModel
from Paper.MyMethod.utils import read_param
import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMCell

class basicLSTM(BaseModel):
    def __init__(self, config):
        super(basicLSTM, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = self.config.is_training
        self.x = tf.placeholder(tf.float64, shape=self.config.x_shape)
        self.y = tf.placeholder(tf.int32, shape=self.config.y_shape)
        self.seqlen = tf.placeholder(tf.int32, shape=self.config.seq_len_shape)

        with tf.name_scope("embedding"):
            ips, ports = read_param('/home/dsk/Dev/tf_traffic/ip_port')
            self.init_portembeds = tf.random_uniform([len(ports), self.config.embedding_size], -1.0, 1.0)
            self.port_embeddings = tf.Variable(self.init_portembeds)
            self.port_input = tf.placeholder(tf.int32, shape=[None])
            self.embed_port_op = tf.nn.embedding_lookup(self.port_embeddings, self.port_input)

        lstm_cell = LSTMCell(name='basic_lstm_cell', num_units=self.config.lstm_hidden, forget_bias=1.0)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, self.x, dtype=tf.float64, sequence_length=self.seqlen)
        print('outputs', outputs)
        print('states', states)
        #states_drop = tf.nn.dropout(states.h, self.config.dropout_rate)
        outputs_drop = tf.nn.dropout(outputs, self.config.dropout_rate)
        """
        For Sequence Labelling
        """
        # outputs.shape ==> (batch_size, n_step, lstm_hidden)
        stacked_rnn_outputs = tf.reshape(outputs_drop, [-1, self.config.lstm_hidden])
        # stacked_rnn_outputs.shape ==> (batch_size*n_step, lstm_hidden)
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, self.config.num_classes, name="dense")
        # stacked_outputs.shape ==> (batch_size*n_step, num_classes)
        #logits = tf.reshape(stacked_outputs, [-1, self.config.n_steps, self.config.num_classes])
        logits = tf.reshape(stacked_outputs, [self.config.batch_size, -1, self.config.num_classes])
        # logits.shape ==> (batch_size, n_step, num_classes)
        #logits = tf.layers.dense(states_drop, self.config.num_classes, name="dense", activation=tf.nn.relu)

        softmax_op = tf.nn.softmax(logits)
        #softmax_op.shape ==> (batch_size, n_step, num_classes)
        #self.predictions = tf.argmax(softmax_op, 1)
        self.predictions = tf.argmax(softmax_op, 2)
        print('predictions', self.predictions)
        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
            print('cross_entropy', self.cross_entropy)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                             global_step=self.global_step_tensor)
            correct_op = tf.equal(tf.cast(self.predictions, tf.int32), self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_op, tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


