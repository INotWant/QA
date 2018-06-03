# coding=utf-8
import math

import paddle.v2 as paddle
import paddle.v2.activation as Act
import paddle.v2.attr as Attr
import paddle.v2.layer as layer

import mLSTM_crf_config


def get_embedding(input, word_vec_dim, wordvecs):
    """
    Define word embedding

    :param input: layer input
    :type input: LayerOutput
    :param word_vec_dim: dimension of the word embeddings
    :type word_vec_dim: int
    :param wordvecs: word embedding matrix
    :type wordvecs: numpy array
    :return: embedding
    :rtype: LayerOutput
    """
    return paddle.layer.embedding(
        input=input,
        size=word_vec_dim,
        param_attr=paddle.attr.ParamAttr(
            name="wordvecs", is_static=True, initializer=lambda _: wordvecs))


class MatchLstm(object):
    """
    Implements Match-LSTM.
    """

    def __init__(self):
        self.name = 'match_lstm'

    def drop_out(self, input, drop_rate=0.5):
        """
        Implements drop out.

        Args:
            input: the LayerOutput needs to apply drop out.
            drop_rate: drop out rate.

        Returns:
            The layer output after applying drop out.
        """
        with layer.mixed(
                layer_attr=Attr.ExtraLayerAttribute(
                    drop_rate=drop_rate),
                bias_attr=False) as dropped:
            dropped += layer.identity_projection(input)
        return dropped

    def fusion_layer(self, input1, input2):
        """
        Combine input1 and input2 by concat(input1 .* input2, input1 - input2,
        input1, input2)
        """
        # fusion layer
        neg_input2 = layer.slope_intercept(input=input2,
                                           slope=-1.0,
                                           intercept=0.0)
        diff1 = layer.addto(input=[input1, neg_input2],
                            act=Act.Identity(),
                            bias_attr=False)
        diff2 = layer.mixed(bias_attr=False,
                            input=layer.dotmul_operator(a=input1, b=input2))

        fused = layer.concat(input=[input1, input2, diff1, diff2])
        return fused

    def get_enc(self, input, conf, type='q'):
        """
        Encodes the input by feeding it into a bidirectional lstm and
        concatenates the forward and backward expression of each time step.
        """
        embs = get_embedding(input, conf.word_vec_dim, conf.wordvecs)
        enc = paddle.networks.bidirectional_lstm(
            input=embs,
            size=conf.word_vec_dim,
            fwd_mat_param_attr=Attr.Param('f_enc_mat.w' + type),
            fwd_bias_param_attr=Attr.Param('f_enc.bias' + type,
                                           initial_std=0.),
            fwd_inner_param_attr=Attr.Param('f_enc_inn.w' + type),
            bwd_mat_param_attr=Attr.Param('b_enc_mat.w' + type),
            bwd_bias_param_attr=Attr.Param('b_enc.bias' + type,
                                           initial_std=0.),
            bwd_inner_param_attr=Attr.Param('b_enc_inn.w' + type),
            return_seq=True)
        enc_dropped = self.drop_out(enc, drop_rate=0.5)
        return enc_dropped

    def _attention(self, direct, cur_token, prev, to_apply, to_apply_proj):
        with layer.mixed(size=cur_token.size,
                         bias_attr=Attr.Param(direct + '.bp',
                                              initial_std=0.),
                         act=Act.Linear()) as proj:
            proj += layer.full_matrix_projection(
                input=cur_token,
                param_attr=Attr.Param(direct + '.wp'))
            proj += layer.full_matrix_projection(
                input=prev,
                param_attr=Attr.Param(direct + '.wr'))

        # 作用：一个0层序列经过运算扩展成一个单层序列，或者一个双层序列
        expanded = layer.expand(input=proj, expand_as=to_apply)
        att_context = layer.addto(input=[expanded, to_apply_proj],
                                  act=Act.Tanh(),
                                  bias_attr=False)

        att_weights = layer.fc(input=att_context,
                               param_attr=Attr.Param(direct + '.w'),
                               bias_attr=Attr.Param(direct + '.b',
                                                    initial_std=0.),
                               act=Act.SequenceSoftmax(),
                               size=1)
        scaled = layer.scaling(input=to_apply, weight=att_weights)
        applied = layer.pooling(input=scaled,
                                pooling_type=paddle.pooling.Sum())
        return applied

    # def _step(self, name, h_q_all, q_proj, h_p_cur):
    #     """
    #     Match-LSTM step. This function performs operations done in one
    #     time step.
    #
    #     Args:
    #         h_p_cur: Current hidden of paragraph encodings: h_i.
    #                  This is the `REAL` input of the group, like
    #                  x_t in normal rnn.
    #         h_q_all: Question encodings.
    #
    #     Returns:
    #         The $h^{r}_{i}$ in the paper.
    #     """
    #     direct = 'left' if 'left' in name else 'right'
    #
    #     # 获取上一个时间步的输出
    #     h_r_prev = paddle.layer.memory(name=name + '_out_',
    #                                    size=h_q_all.size,
    #                                    boot_layer=None)
    #     # h_p_cur :: Current hidden of paragraph encodings
    #     # h_q_all :: q wordEmbedding
    #     # q_proj  :: q_proj_(left or right)
    #     q_expr = self._attention(direct, h_p_cur, h_r_prev, h_q_all, q_proj)
    #     z_cur = self.fusion_layer(h_p_cur, q_expr)
    #
    #     # layer.mixed :: 综合输入映射到指定维度，为 lstm 的输入做准备！
    #     with layer.mixed(size=h_q_all.size * 4,
    #                      act=Act.Tanh(),
    #                      bias_attr=False) as match_input:
    #         match_input += layer.full_matrix_projection(
    #             input=z_cur,
    #             param_attr=Attr.Param('match_input_%s.w0' % direct))
    #
    #     step_out = paddle.networks.lstmemory_unit(
    #         name=name + '_out_',
    #         out_memory=h_r_prev,
    #         param_attr=Attr.Param('step_lstm_%s.w' % direct),
    #         input_proj_bias_attr=Attr.Param(
    #             'step_lstm_mixed_%s.bias' % direct,
    #             initial_std=0.),
    #         lstm_bias_attr=Attr.Param('step_lstm_%s.bias' % direct,
    #                                   initial_std=0.),
    #         input=match_input,
    #         size=h_q_all.size)
    #     return step_out

    def _step(self, name, h_q_all, q_proj, h_p_cur, qe_comm, ee_comm):
        """
        Match-LSTM step. This function performs operations done in one
        time step.

        Args:
            h_p_cur: Current hidden of paragraph encodings: h_i.
                     This is the `REAL` input of the group, like
                     x_t in normal rnn.
            h_q_all: Question encodings.

        Returns:
            The $h^{r}_{i}$ in the paper.
        """
        conf = mLSTM_crf_config.TrainingConfig()
        direct = 'left' if 'left' in name else 'right'

        # 获取上一个时间步的输出
        h_r_prev = paddle.layer.memory(name=name + '_out_',
                                       size=h_q_all.size,
                                       boot_layer=None)
        # h_p_cur :: Current hidden of paragraph encodings
        # h_q_all :: q wordEmbedding
        # q_proj  :: q_proj_(left or right)
        q_expr = self._attention(direct, h_p_cur, h_r_prev, h_q_all, q_proj)
        z_cur = self.fusion_layer(h_p_cur, q_expr)

        # feature embeddings
        comm_initial_std = 1 / math.sqrt(64.0)
        qe_comm_emb = paddle.layer.embedding(
            input=qe_comm,
            size=conf.com_vec_dim,
            param_attr=paddle.attr.ParamAttr(
                name="_cw_embedding.w0",
                initial_std=comm_initial_std,
                l2_rate=conf.default_l2_rate))

        ee_comm_emb = paddle.layer.embedding(
            input=ee_comm,
            size=conf.com_vec_dim,
            param_attr=paddle.attr.ParamAttr(
                name="_eecom_embedding.w0",
                initial_std=comm_initial_std,
                l2_rate=conf.default_l2_rate))

        # layer.mixed :: 综合输入映射到指定维度，为 lstm 的输入做准备！
        with layer.mixed(size=h_q_all.size * 4,
                         act=Act.Tanh(),
                         bias_attr=False) as match_input:
            match_input += layer.full_matrix_projection(
                input=z_cur,
                param_attr=Attr.Param('match_input_z_%s.w0' % direct))
            match_input += layer.full_matrix_projection(
                input=qe_comm_emb,
                param_attr=Attr.Param('match_input_qe_%s.w0' % direct))
            match_input += layer.full_matrix_projection(
                input=ee_comm_emb,
                param_attr=Attr.Param('match_input_ee_%s.w0' % direct))

        step_out = paddle.networks.lstmemory_unit(
            name=name + '_out_',
            out_memory=h_r_prev,
            param_attr=Attr.Param('step_lstm_%s.w' % direct),
            input_proj_bias_attr=Attr.Param(
                'step_lstm_mixed_%s.bias' % direct,
                initial_std=0.),
            lstm_bias_attr=Attr.Param('step_lstm_%s.bias' % direct,
                                      initial_std=0.),
            input=match_input,
            size=h_q_all.size)
        return step_out

    def recurrent_group(self, name, inputs, reverse=False):
        """
        Implements the Match-LSTM layer in the paper.

        Args:
            name: the name prefix of the layers created by this method.
            inputs: the inputs takes by the _step method.
            reverse: True if the paragraph encoding is processed from right
                     to left, otherwise the paragraph encoding is processed
                     from left to right.
        Returns:
            The Match-LSTM layer's output of one direction.
        """
        inputs.insert(0, name)
        seq_out = layer.recurrent_group(name=name,
                                        input=inputs,
                                        step=self._step,
                                        reverse=reverse)
        return seq_out

    def network(self, question, evidence, conf):
        """
        Implements the whole network of Match-LSTM.

        Returns:
            A tuple of LayerOutput objects containing the start and end
            probability distributions respectively.
        """

        q_enc = self.get_enc(question, conf, type='q')
        p_enc = self.get_enc(evidence, conf, type='q')

        q_proj_left = layer.fc(size=conf.word_vec_dim * 2,
                               bias_attr=False,
                               param_attr=Attr.Param(
                                   self.name + '_left_' + '.wq'),
                               input=q_enc)
        q_proj_right = layer.fc(size=conf.word_vec_dim * 2,
                                bias_attr=False,
                                param_attr=Attr.Param(
                                    self.name + '_right_' + '.wq'),
                                input=q_enc)
        # StaticInput 定义了一个只读的Memory，由StaticInput指定的输入不会被recurrent_group拆解，
        # recurrent_group 循环展开的每个时间步总是能够引用所有输入，可以是一个非序列，或者一个单层序列。
        left_out = self.recurrent_group(
            self.name + '_left',
            [layer.StaticInput(q_enc),
             layer.StaticInput(q_proj_left), p_enc],
            reverse=False)
        right_out = self.recurrent_group(
            self.name + '_right_',
            [layer.StaticInput(q_enc),
             layer.StaticInput(q_proj_right), p_enc],
            reverse=True)
        match_seq = layer.concat(input=[left_out, right_out])
        return self.drop_out(match_seq, drop_rate=0.5)

    def network(self, question, evidence, qe_comm, ee_comm, conf):
        """
        Implements the whole network of Match-LSTM.

        Returns:
            A tuple of LayerOutput objects containing the start and end
            probability distributions respectively.
        """

        q_enc = self.get_enc(question, conf, type='q')
        p_enc = self.get_enc(evidence, conf, type='q')

        q_proj_left = layer.fc(size=conf.word_vec_dim * 2,
                               bias_attr=False,
                               param_attr=Attr.Param(
                                   self.name + '_left_' + '.wq'),
                               input=q_enc)
        q_proj_right = layer.fc(size=conf.word_vec_dim * 2,
                                bias_attr=False,
                                param_attr=Attr.Param(
                                    self.name + '_right_' + '.wq'),
                                input=q_enc)
        # StaticInput 定义了一个只读的Memory，由StaticInput指定的输入不会被recurrent_group拆解，
        # recurrent_group 循环展开的每个时间步总是能够引用所有输入，可以是一个非序列，或者一个单层序列。
        left_out = self.recurrent_group(
            self.name + '_left',
            [layer.StaticInput(q_enc),
             layer.StaticInput(q_proj_left), p_enc, qe_comm, ee_comm],
            reverse=False)
        right_out = self.recurrent_group(
            self.name + '_right_',
            [layer.StaticInput(q_enc),
             layer.StaticInput(q_proj_right), p_enc, qe_comm, ee_comm],
            reverse=True)
        match_seq = layer.concat(input=[left_out, right_out])
        return self.drop_out(match_seq, drop_rate=0.5)
