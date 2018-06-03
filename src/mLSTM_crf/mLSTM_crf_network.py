# coding=utf-8
import os
import sys

import paddle.v2 as paddle

from match_LSTM import MatchLstm

# 把目录加入环境变量
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import reader

__all__ = ["training_net", "inference_net", "feeding"]

feeding = {
    reader.Q_IDS_STR: reader.Q_IDS,
    reader.E_IDS_STR: reader.E_IDS,
    reader.QE_COMM_STR: reader.QE_COMM,
    reader.EE_COMM_STR: reader.EE_COMM,
    reader.LABELS_STR: reader.LABELS
}


def training_net(conf):
    """
    Define training network

    :param conf: network conf
    :return: CRF cost
    :rtype: LayerOutput
    """
    e_encoding, label = define_common_network(conf)
    crf = paddle.layer.crf(
        input=e_encoding,
        label=label,
        size=conf.label_num,
        param_attr=paddle.attr.ParamAttr(
            name="_crf.w0",
            initial_std=conf.default_init_std,
            l2_rate=conf.default_l2_rate),
        layer_attr=paddle.attr.ExtraAttr(device=-1))

    return crf


# def define_data(dict_dim, label_num):
#     """
#     Define data layers
#
#     :param dict_dim: number of words in the vocabulary
#     :type dict_dim: int
#     :param label_num: label numbers, BIO:3, BIO2:4
#     :type label_num: int
#     :return: data layers
#     :rtype: tuple of LayerOutput
#     """
#     question = paddle.layer.data(
#         name=reader.Q_IDS_STR,
#         type=paddle.data_type.integer_value_sequence(dict_dim))
#
#     evidence = paddle.layer.data(
#         name=reader.E_IDS_STR,
#         type=paddle.data_type.integer_value_sequence(dict_dim))
#
#     label = paddle.layer.data(
#         name=reader.LABELS_STR,
#         type=paddle.data_type.integer_value_sequence(label_num),
#         layer_attr=paddle.attr.ExtraAttr(device=-1))
#
#     return question, evidence, label


def define_data(dict_dim, label_num):
    """
    Define data layers

    :param dict_dim: number of words in the vocabulary
    :type dict_dim: int
    :param label_num: label numbers, BIO:3, BIO2:4
    :type label_num: int
    :return: data layers
    :rtype: tuple of LayerOutput
    """
    question = paddle.layer.data(
        name=reader.Q_IDS_STR,
        type=paddle.data_type.integer_value_sequence(dict_dim))

    evidence = paddle.layer.data(
        name=reader.E_IDS_STR,
        type=paddle.data_type.integer_value_sequence(dict_dim))

    qe_comm = paddle.layer.data(
        name=reader.QE_COMM_STR,
        type=paddle.data_type.integer_value_sequence(2))

    ee_comm = paddle.layer.data(
        name=reader.EE_COMM_STR,
        type=paddle.data_type.integer_value_sequence(2))

    label = paddle.layer.data(
        name=reader.LABELS_STR,
        type=paddle.data_type.integer_value_sequence(label_num),
        layer_attr=paddle.attr.ExtraAttr(device=-1))

    return question, evidence, qe_comm, ee_comm, label


def define_common_network(conf):
    """
        Define common network

        :param conf: network conf
        :return: CRF features, golden labels
        :rtype: tuple
        """
    # define data layers
    # question, evidence, label = \
    #     define_data(conf.dict_dim, conf.label_num)

    question, evidence, qe_comm, ee_comm, label = \
        define_data(conf.dict_dim, conf.label_num)

    # mlstm_encoding = MatchLstm().network(question, evidence, conf)

    mlstm_encoding = MatchLstm().network(question, evidence, qe_comm, ee_comm, conf)

    crf_feats = paddle.layer.fc(
        act=paddle.activation.Linear(),
        input=mlstm_encoding,
        size=conf.label_num,
        param_attr=paddle.attr.ParamAttr(
            name="_output.w0",
            initial_std=conf.default_init_std,
            l2_rate=conf.default_l2_rate),
        bias_attr=False)
    return crf_feats, label


def inference_net(conf):
    """
    Define training network

    :param conf: network conf
    :return: CRF viberbi decoding result
    :rtype: LayerOutput
    """
    e_encoding, label = define_common_network(conf)
    ret = paddle.layer.crf_decoding(
        input=e_encoding,
        size=conf.label_num,
        param_attr=paddle.attr.ParamAttr(name="_crf.w0"),
        layer_attr=paddle.attr.ExtraAttr(device=-1))

    return ret
