#! -*- coding: utf-8 -*-
# NLI微调预训练模型

from utils import *
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from keras.layers import Input, Dense, Lambda
from keras.initializers import TruncatedNormal
import sys

# 基本参数
model_type = sys.argv[1]
epochs = 1
learning_rate = 2e-5
batch_size = 16
warmup_proportion = 0.1

# bert配置
assert model_type in ['base', 'large']
if model_type == 'base':
    model_name = 'uncased_L-12_H-768_A-12'
elif model_type == 'large':
    model_name = 'uncased_L-24_H-1024_A-16'

config_path = '/root/kg/bert/%s/bert_config.json' % model_name
checkpoint_path = '/root/kg/bert/%s/bert_model.ckpt' % model_name
dict_path = '/root/kg/bert/%s/vocab.txt' % model_name

# 加载NLI
data_path = '/root/senteval/'
mnli_train = load_mnli_train_data(data_path + 'MNLI/train.tsv')
snli_train = load_snli_data(data_path + 'SNLI/train')
snli_dev = load_snli_data(data_path + 'SNLI/dev')
snli_test = load_snli_data(data_path + 'SNLI/test')
nli_data = mnli_train + snli_train + snli_dev + snli_test

# 建立分词器
tokenizer = get_tokenizer(dict_path)

# 数据转换为ID
a_token_ids, b_token_ids, labels = convert_to_ids(nli_data, tokenizer)
label2id = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
labels = np.array([[label2id[l]] for l in labels])


def merge(inputs):
    """向量合并：a、b、|a-b|拼接
    """
    a, b = inputs
    o = K.concatenate([a, b, K.abs(a - b)], axis=1)
    return o


# 建立模型
encoder = get_encoder(config_path, checkpoint_path, 'last-avg')

t1_in = Input(shape=(None,))
t2_in = Input(shape=(None,))
s1_in = Input(shape=(None,))
s2_in = Input(shape=(None,))

z1 = encoder([t1_in, s1_in])
z2 = encoder([t2_in, s2_in])
z = Lambda(merge)([z1, z2])
p = Dense(
    units=3,
    activation='softmax',
    use_bias=False,
    kernel_initializer=TruncatedNormal(stddev=0.02)
)(z)

train_model = Model([t1_in, t2_in, s1_in, s2_in], p)
train_model.summary()

AdamW = extend_with_weight_decay(Adam, name='AdamW')
AdamWLR = extend_with_piecewise_linear_lr(AdamW, name='AdamWLR')
train_steps = int(len(nli_data) / batch_size * epochs)
warmup_steps = int(train_steps * warmup_proportion)
optimizer = AdamWLR(
    learning_rate=learning_rate,
    weight_decay_rate=0.01,
    exclude_from_weight_decay=['Norm', 'bias'],
    lr_schedule={warmup_steps: 1, train_steps: 0}
)
train_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

a_segment_ids = np.zeros_like(a_token_ids)
b_segment_ids = np.zeros_like(b_token_ids)

# 训练模型
train_model.fit([a_token_ids, b_token_ids, a_segment_ids, b_segment_ids],
                labels,
                epochs=epochs,
                batch_size=batch_size)

# 保存模型
encoder.save_weights('weights/' + model_name + '_nli.weights')
