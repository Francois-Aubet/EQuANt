import tensorflow as tf
from layers import *




class Model(object):
    """
    The class defining the model.
    The two main functions are the initialization of the model: creating placeholders and such
    and the function defining the forward pass.
    """


    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True, graph = None):
        self.config = config

        self._opt = opt

        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():

            # create the variable that countains the count of current epoch
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            # create the placeholder for dropout
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")

            # create the placeholders that get things from the dataset
            self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.y3, self.qa_id = batch.get_next()
            # self.c is the placeholder for the context
            # self.q is the placeholder for the question
            # self.ch is the placeholder for the context chars
            # self.qh is the placeholder for the question chars
            # self.y1 is the placeholder for the true start of the answer
            # self.y2 is the placeholder for the true end of the answer
            # self.y2 is the placeholder for if there is an answer or not
            # self.qa_id is the question id

            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
                word_mat, dtype=tf.float32), trainable=False)

            self.char_mat = tf.get_variable(
                "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

            self.c_mask = tf.cast(self.c, tf.bool)
            self.q_mask = tf.cast(self.q, tf.bool)
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)


            if opt:
                N, CL = config.batch_size, config.char_limit
                self.c_maxlen = tf.reduce_max(self.c_len)
                self.q_maxlen = tf.reduce_max(self.q_len)
                self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
                self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
                self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
                self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
                self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
                self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
                self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
                self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
                self.y3 = tf.reshape(self.y3, [N,])
            else:
                self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit
            
            self.ans_max_len = 600
            self.ans_max_mask = tf.cast(tf.ones([self.ans_max_len]), tf.bool)

            self.ch_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])

            self.qh_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

            # create the graph for the forward pass:
            self.forward()
            # print the total number of parameters in model:
            total_params()

            if trainable:
                # define function to addapt learning rate:
                self.lr = tf.minimum(config.learning_rate, 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
                # define optimizer
                self.train_op = tf.train.AdamOptimizer(learning_rate = self.lr, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7).minimize(self.loss, global_step=self.global_step)



    def forward(self):
        """ Creation of the graph for one forward pass.
        The code is organized as in the paper, in the five layers.
        The exact implementation of the layers is done in the 'layer.py' file. Here the function are only called
        showing a high level representation of the graph.
        """
        config = self.config

        # reading the values from the config
        N = config.batch_size
        PL = self.c_maxlen
        QL = self.q_maxlen
        CL = config.char_limit
        d = config.hidden
        dc = config.char_dim
        nh = config.num_heads


        with tf.variable_scope("Input_Embedding_Layer"):

            ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.ch), [N * PL, CL, dc])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.qh), [N * QL, CL, dc])

            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

			# Bidaf style conv-highway encoder
            ch_emb = conv(ch_emb, d,bias = True, activation = tf.nn.relu, kernel_size = 5, name = "char_conv", reuse = None)
            qh_emb = conv(qh_emb, d,bias = True, activation = tf.nn.relu, kernel_size = 5, name = "char_conv", reuse = True)

            ch_emb = tf.reduce_max(ch_emb, axis = 1)
            qh_emb = tf.reduce_max(qh_emb, axis = 1)

            ch_emb = tf.reshape(ch_emb, [N, PL, ch_emb.shape[-1]])
            qh_emb = tf.reshape(qh_emb, [N, QL, ch_emb.shape[-1]])

            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            c_emb = highway(c_emb, size = d, scope = "highway", dropout = self.dropout, reuse = None)
            q_emb = highway(q_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)



        with tf.variable_scope("Embedding_Encoder_Layer"):
            c = residual_block(c_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.c_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.c_len,
                scope = "Encoder_Residual_Block",
                bias = False,
                dropout = self.dropout)
            q = residual_block(q_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.q_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.q_len,
                scope = "Encoder_Residual_Block",
                reuse = True, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)


        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            if self._opt:
                S = optimized_trilinear_for_attention([c, q], self.c_maxlen, self.q_maxlen, input_keep_prob = 1.0 - self.dropout)
            else:
                C = tf.tile(tf.expand_dims(c,2),[1,1,self.q_maxlen,1])
                Q = tf.tile(tf.expand_dims(q,1),[1,self.c_maxlen,1,1])
                S = trilinear([C, Q, C*Q], input_keep_prob = 1.0 - self.dropout)
            #
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), dim = 1),(0,2,1))
            self.c2q = tf.matmul(S_, q)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]


        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat(attention_outputs, axis = -1)
            self.enc = [conv(inputs, d, name = "input_projection")]
            for i in range(3):
                if i % 2 == 0: # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
                self.enc.append(
                    residual_block(self.enc[i],
                        num_blocks = 7,
                        num_conv_layers = 2,
                        kernel_size = 5,
                        mask = self.c_mask,
                        num_filters = d,
                        num_heads = nh,
                        seq_len = self.c_len,
                        scope = "Model_Encoder",
                        bias = False,
                        reuse = True if i > 0 else None,
                        dropout = self.dropout)
                    )


        # EQUANET3:
        with tf.variable_scope("Answerability_Prediction"):
            # prediction of the unansweribility

            outputs = residual_block(self.enc[1], num_blocks = 2, num_conv_layers = 2,
                        kernel_size = 5, mask = self.c_mask, num_filters = d, num_heads = nh,
                        seq_len = self.c_len, scope = "Model_Encoder", bias = False,
                        reuse = False, dropout = self.dropout)

            dim_chan = 2

            outputs = layer_norm(outputs, scope="nomalizing1")
            outputs = tf.nn.dropout(outputs, 1.0 - self.dropout)
            outputs = conv(outputs, d/2, True, tf.nn.relu, name = "FFN_1", reuse = False)

            outputs = conv(outputs, d/4, True, tf.nn.relu, name = "FFN_3", reuse = False)

            #outputs = layer_norm(outputs, scope="nomalizing4")
            outputs = conv(outputs, 1, True, None, name = "FFN_2", reuse = False)#tf.squeeze(, -1)

            tmp_output = tf.reduce_mean(outputs, axis=1)

            self.logging2 = tmp_output
            self.logging = tmp_output

            self.y3_logit = tmp_output

            self.y3_logit = tf.reshape(self.y3_logit,[N,])

            losses_y3 = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y3, logits=self.y3_logit)

            self.yp3 = tf.nn.sigmoid(self.y3_logit)





        with tf.variable_scope("Output_Layer"):
            # computes the output from the network

            # Paper: 
            start_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[2]],axis = -1),1, bias = False, name = "start_pointer"),-1)
            end_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[3]],axis = -1),1, bias = False, name = "end_pointer"), -1)
            
            self.logits = [mask_logits(start_logits, mask = self.c_mask), mask_logits(end_logits, mask = self.c_mask)]

            logits1, logits2 = [l for l in self.logits]

            # Paper: At inference time, the predicted span (s, e) is chosen such that p 1 s p 2 e is maximized
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, config.ans_limit)


            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits1, labels=self.y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2, labels=self.y2)

            final_loss_function = losses_y3 + self.y3 * (losses + losses2)
            self.loss = tf.reduce_mean(final_loss_function)




        with tf.variable_scope("Attention_monitoring"):
            # prediction of the unansweribility

            self.max_att1 = tf.reduce_max(tf.nn.softmax(logits1))
            self.max_att2 = tf.reduce_max(tf.nn.softmax(logits2))


        if config.l2_norm is not None:
            """ A possibility for regularization. """
            variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
            self.loss += l2_loss

        if config.decay is not None:
            """ A possibility for weight decay / using moving average of the parameters. """
            self.var_ema = tf.train.ExponentialMovingAverage(config.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)

                self.assign_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v:
                        self.assign_vars.append(tf.assign(var,v))


    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step


