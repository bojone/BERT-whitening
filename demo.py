#! -*- coding: utf-8 -*-
# 简单的线性变换（白化）操作，就可以达到BERT-flow的效果。
# 测试任务：GLUE的STS-B。
# 测试环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.9.7

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import open, sequence_padding
from keras.models import Model


def load_train_data(filename):
    """加载训练数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            if i > 0:
                l = l.strip().split('\t')
                D.append((l[-3], l[-2], float(l[-1])))
    return D


def load_test_data(filename):
    """加载测试数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = l.strip().split('\t')
            D.append((l[-2], l[-1], float(l[-3])))
    return D


# 加载数据集
train_data = load_train_data('/root/glue/STS-B/train.tsv')
test_data = load_test_data('/root/glue/STS-B/sts-test.csv')

# bert配置
config_path = '/root/kg/bert/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/uncased_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器


class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    """自定义全局池化
    """
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())[:, :, None]
            return K.sum(inputs * mask, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(inputs, axis=1)


# 建立模型
bert = build_transformer_model(config_path, checkpoint_path)

encoder_layers, count = [], 0
while True:
    try:
        output = bert.get_layer(
            'Transformer-%d-FeedForward-Norm' % count
        ).output
        encoder_layers.append(output)
        count += 1
    except:
        break

n_last, outputs = 2, []
for i in range(n_last):
    outputs.append(GlobalAveragePooling1D()(encoder_layers[-i]))

output = keras.layers.Average()(outputs)

# 最后的编码器
encoder = Model(bert.inputs, output)


def convert_to_vecs(data, maxlen=64):
    """转换文本数据为id形式
    """
    a_token_ids, b_token_ids, labels = [], [], []
    for d in data:
        token_ids = tokenizer.encode(d[0], maxlen=maxlen)[0]
        a_token_ids.append(token_ids)
        token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
        b_token_ids.append(token_ids)
        labels.append(d[2])
    a_token_ids = sequence_padding(a_token_ids)
    b_token_ids = sequence_padding(b_token_ids)
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
    W = np.dot(np.diag(s**0.5), vh)
    W = np.linalg.inv(W)
    return W, -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


# 语料向量化，计算变换矩阵和偏置项
a_train_vecs, b_train_vecs, train_labels = convert_to_vecs(train_data)
a_test_vecs, b_test_vecs, test_labels = convert_to_vecs(test_data)
kernel, bias = compute_kernel_bias([
    a_train_vecs, b_train_vecs, a_test_vecs, b_test_vecs
])

# 变换，标准化，相似度
a_train_vecs = transform_and_normalize(a_train_vecs, kernel, bias)
b_train_vecs = transform_and_normalize(b_train_vecs, kernel, bias)
train_sims = (a_train_vecs * b_train_vecs).sum(axis=1)
print(u'训练集的相关系数：%s' % np.corrcoef(train_labels, train_sims)[0, 1])

# 变换，标准化，相似度
a_test_vecs = transform_and_normalize(a_test_vecs, kernel, bias)
b_test_vecs = transform_and_normalize(b_test_vecs, kernel, bias)
test_sims = (a_test_vecs * b_test_vecs).sum(axis=1)
print(u'测试集的相关系数：%s' % np.corrcoef(test_labels, test_sims)[0, 1])
