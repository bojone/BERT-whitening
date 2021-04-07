#! -*- coding: utf-8 -*-
# 数据读取函数

import os
from tqdm import tqdm
import numpy as np
import scipy.stats
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import open
from bert4keras.snippets import sequence_padding
from keras.layers import GlobalAveragePooling1D
from keras.models import Model

if not os.path.exists('weights'):
    os.mkdir('weights')


def load_sts_b_train_data(filename):
    """加载STS-B训练数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            if i > 0:
                l = l.strip().split('\t')
                D.append((l[-3], l[-2], float(l[-1])))
    return D


def load_sts_b_test_data(filename):
    """加载STS-B测试数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = l.strip().split('\t')
            D.append((l[-2], l[-1], float(l[-3])))
    return D


def load_sts_12_16_data(filename):
    """加载STS-12,13,14,15,16数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    input_file = filename
    label_file = input_file.replace('STS.input', 'STS.gs')
    input_file = open(input_file, encoding='utf-8')
    label_file = open(label_file, encoding='utf-8')
    for i, l in zip(input_file, label_file):
        if l.strip():
            i = i.strip().split('\t')
            l = float(l.strip())
            D.append((i[0], i[1], l))
    input_file.close()
    label_file.close()
    return D


def load_sick_r_data(filename):
    """加载SICK-R数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            if i > 0:
                l = l.strip().split('\t')
                D.append((l[1], l[2], float(l[3])))
    return D


def load_mnli_train_data(filename):
    """加载MNLI训练数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            if i > 0:
                l = l.strip().split('\t')
                D.append((l[8], l[9], l[10]))
    return D


def load_snli_data(filename):
    """加载SNLI数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    filename = filename.split('/')
    s1_file = '/'.join(filename[:-1]) + '/s1.' + filename[-1]
    s2_file = '/'.join(filename[:-1]) + '/s2.' + filename[-1]
    l_file = '/'.join(filename[:-1]) + '/labels.' + filename[-1]
    s1_file = open(s1_file, encoding='utf-8')
    s2_file = open(s2_file, encoding='utf-8')
    l_file = open(l_file, encoding='utf-8')
    for s1, s2, l in zip(s1_file, s2_file, l_file):
        D.append((s1.strip(), s2.strip(), l.strip()))
    s1_file.close()
    s2_file.close()
    l_file.close()
    return D


sts_12_16_names = {
    'STS-12': [
        'MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', 'surprise.SMTnews'
    ],
    'STS-13': ['FNWN', 'headlines', 'OnWN'],
    'STS-14': [
        'deft-forum', 'deft-news', 'headlines', 'images', 'OnWN', 'tweet-news'
    ],
    'STS-15': [
        'answers-forums', 'answers-students', 'belief', 'headlines', 'images'
    ],
    'STS-16': [
        'answer-answer', 'headlines', 'plagiarism', 'postediting',
        'question-question'
    ]
}


def get_tokenizer(dict_path):
    """建立分词器
    """
    return Tokenizer(dict_path, do_lower_case=True)


def get_encoder(config_path, checkpoint_path, pooling='first-last-avg'):
    """建立编码器
    """
    assert pooling in ['first-last-avg', 'last-avg']

    bert = build_transformer_model(config_path, checkpoint_path)

    outputs, count = [], 0
    while True:
        try:
            output = bert.get_layer(
                'Transformer-%d-FeedForward-Norm' % count
            ).output
            outputs.append(output)
            count += 1
        except:
            break

    if pooling == 'first-last-avg':
        outputs = [
            keras.layers.GlobalAveragePooling1D()(outputs[0]),
            keras.layers.GlobalAveragePooling1D()(outputs[-1])
        ]
        output = keras.layers.Average()(outputs)
    elif pooling == 'last-avg':
        output = keras.layers.GlobalAveragePooling1D()(outputs[-1])

    # 最后的编码器
    encoder = Model(bert.inputs, output)
    return encoder


def convert_to_ids(data, tokenizer, maxlen=64):
    """转换文本数据为id形式
    """
    a_token_ids, b_token_ids, labels = [], [], []
    for d in tqdm(data):
        token_ids = tokenizer.encode(d[0], maxlen=maxlen)[0]
        a_token_ids.append(token_ids)
        token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
        b_token_ids.append(token_ids)
        labels.append(d[2])
    a_token_ids = sequence_padding(a_token_ids)
    b_token_ids = sequence_padding(b_token_ids)
    return a_token_ids, b_token_ids, labels


def convert_to_vecs(data, tokenizer, encoder, maxlen=64):
    """转换文本数据为向量形式
    """
    a_token_ids, b_token_ids, labels = convert_to_ids(data, tokenizer, maxlen)
    a_vecs = encoder.predict([a_token_ids,
                              np.zeros_like(a_token_ids)],
                             verbose=True)
    b_vecs = encoder.predict([b_token_ids,
                              np.zeros_like(b_token_ids)],
                             verbose=True)
    return a_vecs, b_vecs, np.array(labels)


def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation
