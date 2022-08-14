from base import BaseModel
from Paper.MyMethod.utils import read_param
import tensorflow as tf
from tensorflow.core.rnn_cell import LSTMCell

class basicLSTM_bidir(BaseModel):
    def __init__(self, config):
        super(basicLSTM_bidir, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = self.config.is_training
        self.x = tf.placeholder(tf.float64, shape=self.config.x_shape)
        self.y = tf.placeholder(tf.int32, shape=self.config.y_shape)
        self.seqlen = tf.placeholder(tf.int32, shape=self.config.seq_len_shape)

        with tf.name_scope("basicLSTM_bidir/embedding"):
            ips, ports = read_param('/home/dsk/Dev/tf_traffic/ip_port')
            #ip --> 1857, port --> 599
            # 生成shape=[端口数目，embedding_size]的张量，将端口号转换成一个向量，张量的值是[-1,1]均匀分布的随机数
            self.init_portembeds = tf.random_uniform([len(ports), self.config.embedding_size], -1.0, 1.0)
            #用init_portembeds的值初始化变量port_embeddings
            self.port_embeddings = tf.Variable(self.init_portembeds)
            self.port_input = tf.placeholder(tf.int32, shape=[None])
            #将每个端口值转换成对应的embedding
            self.embed_port_op = tf.nn.embedding_lookup(self.port_embeddings, self.port_input)
        #定义了正向和反向LSTM
        fw_lstm_cell = LSTMCell(name='basicLSTM_bidir/fw_lstm_cell', num_units=self.config.lstm_hidden, forget_bias=1.0)
        bw_lstm_cell = LSTMCell(name='basicLSTM_bidir/bw_lstm_cell', num_units=self.config.lstm_hidden, forget_bias=1.0)

        #outputs, states = tf.nn.dynamic_rnn(lstm_cell, self.x, dtype=tf.float64, sequence_length=self.seqlen)
        #将数据x送入双向LSTM进行训练，
        # 输出：outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的元组,所有time_step的输出
        #output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的元组,最后一个time_step的输出
        outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, self.x, dtype=tf.float64, sequence_length=self.seqlen)
        output_fw, output_bw = outputs
        self.final_outputs = tf.concat((output_fw, output_bw),2)#将第2维拼接在一起，
        # outputs = (output_fw, output_bw)
        # output_fw.shape (batch_size, n_step, cell_fw.output_size)
        # output_bw.shape (batch_size, n_step, cell_bw.output_size)
        print('outputs', outputs)
        print('states', states)
        #states_drop = tf.nn.dropout(states.h, self.config.dropout_rate)
        final_outputs_drop = tf.nn.dropout(self.final_outputs, self.config.dropout_rate)
        """
        For Sequence Labelling
        """
        # outputs.shape ==> (batch_size, n_step, lstm_hidden*2)
        stacked_rnn_outputs = tf.reshape(final_outputs_drop, [-1, self.config.lstm_hidden*2])
        # stacked_rnn_outputs.shape ==> (batch_size*n_step, lstm_hidden*2)
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, self.config.num_classes, name="basicLSTM_bidir/dense")
        # stacked_outputs.shape ==> (batch_size*n_step, num_classes)
        #logits = tf.reshape(stacked_outputs, [-1, self.config.n_steps, self.config.num_classes])
        self.logits = tf.reshape(stacked_outputs, [self.config.batch_size, -1, self.config.num_classes])
        # logits.shape ==> (batch_size, n_step, num_classes)
        #logits = tf.layers.dense(states_drop, self.config.num_classes, name="dense", activation=tf.nn.relu)

        self.softmax_op = tf.nn.softmax(self.logits)
        #softmax_op.shape ==> (batch_size, n_step, num_classes)
        #self.predictions = tf.argmax(softmax_op, 1)
        self.predictions = tf.argmax(self.softmax_op, 2)
        print('predictions', self.predictions)
        with tf.name_scope("basicLSTM_bidir/loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
            print('cross_entropy', self.cross_entropy)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                             global_step=self.global_step_tensor)
            correct_op = tf.equal(tf.cast(self.predictions, tf.int32), self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_op, tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


