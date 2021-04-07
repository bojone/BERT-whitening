#! -*- coding: utf-8 -*-
# 简单的线性变换（白化）操作，就可以达到甚至超过BERT-flow的效果。

from utils import *
import os, sys

# 基本参数
model_type, task_name, n_components, normalized_by = sys.argv[1:]
assert model_type in ['base', 'large', 'base-nli', 'large-nli']
assert task_name in [
    'STS-B', 'STS-12', 'STS-13', 'STS-14', 'STS-15', 'STS-16', 'SICK-R'
]
assert normalized_by in ['target', 'nli']

n_components = int(n_components)
if n_components < 0:
    if model_type.startswith('base'):
        n_components = 768
    elif model_type.startswith('large'):
        n_components = 1024

# 加载数据集
data_path = '/root/senteval/'

if task_name == 'STS-B':
    datasets = {
        'sts-b-train': load_sts_b_train_data(data_path + 'STS-B/train.tsv'),
        'sts-b-dev': load_sts_b_train_data(data_path + 'STS-B/dev.tsv'),
        'sts-b-test': load_sts_b_test_data('/root/glue/STS-B/sts-test.csv')
    }
elif task_name.startswith('STS-1'):
    names = sts_12_16_names[task_name]
    datasets = {
        n: load_sts_12_16_data(data_path + task_name + '/STS.input.%s.txt' % n)
        for n in names
    }
elif task_name == 'SICK-R':
    datasets = {
        'sick-r-train': load_sick_r_data(data_path + 'SICK-R/SICK_train.txt'),
        'sick-r-dev': load_sick_r_data(data_path + 'SICK-R/SICK_trial.txt'),
        'sick-r-test':
            load_sick_r_data(data_path + 'SICK-R/SICK_test_annotated.txt'),
    }

# bert配置
if model_type.startswith('base'):
    model_name = 'uncased_L-12_H-768_A-12'
elif model_type.startswith('large'):
    model_name = 'uncased_L-24_H-1024_A-16'

config_path = '/root/kg/bert/%s/bert_config.json' % model_name
checkpoint_path = '/root/kg/bert/%s/bert_model.ckpt' % model_name
dict_path = '/root/kg/bert/%s/vocab.txt' % model_name

# 建立分词器
tokenizer = get_tokenizer(dict_path)

# 建立模型
encoder = get_encoder(config_path, checkpoint_path)

# 加载NLI预训练权重
if model_type.endswith('-nli'):
    encoder.load_weights('weights/' + model_name + '_nli.weights')

# 语料向量化
all_names, all_weights, all_vecs, all_labels = [], [], [], []
for name, data in datasets.items():
    a_vecs, b_vecs, labels = convert_to_vecs(data, tokenizer, encoder)
    all_names.append(name)
    all_weights.append(len(data))
    all_vecs.append((a_vecs, b_vecs))
    all_labels.append(labels)

# 计算变换矩阵和偏置项
if n_components == 0:
    kernel, bias = None, None
else:
    if normalized_by == 'target':
        kernel, bias = compute_kernel_bias([
            v for vecs in all_vecs for v in vecs
        ])
    elif normalized_by == 'nli':
        if os.path.exists('weights/nli.%s.kernel.bias.npy' % model_type):
            kernel, bias = np.load(
                'weights/nli.%s.kernel.bias.npy' % model_type,
                allow_pickle=True
            )
        else:
            mnli_train = load_mnli_train_data(data_path + 'MNLI/train.tsv')
            snli_train = load_snli_data(data_path + 'SNLI/train')
            snli_dev = load_snli_data(data_path + 'SNLI/dev')
            snli_test = load_snli_data(data_path + 'SNLI/test')
            nli_data = mnli_train + snli_train + snli_dev + snli_test
            nli_a_vecs, nli_b_vecs, _ = convert_to_vecs(
                nli_data, tokenizer, encoder
            )
            kernel, bias = compute_kernel_bias([nli_a_vecs, nli_b_vecs])
            np.save('weights/nli.%s.kernel.bias' % model_type, [kernel, bias])
    kernel = kernel[:, :n_components]

# 变换，标准化，相似度，相关系数
all_corrcoefs = []
for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
    a_vecs = transform_and_normalize(a_vecs, kernel, bias)
    b_vecs = transform_and_normalize(b_vecs, kernel, bias)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(labels, sims)
    all_corrcoefs.append(corrcoef)

all_corrcoefs.extend([
    np.average(all_corrcoefs),
    np.average(all_corrcoefs, weights=all_weights)
])

for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
    print('%s: %s' % (name, corrcoef))
