# BERT-whitening 英文测试

BERT-whitening在常见英文数据集上的测试，基本上已经对齐了BERT-flow的设置。

## 文件

```
- utils.py  工具函数
- eval.py  评测主文件
- nli_train.py  用NLI数据微调模型（即Sentence-BERT的Keras版）
```

## 评测

命令格式：
```
python eval.py [model_type] [task_name] [n_components] [normalized_by]
```

使用例子：
```
python eval.py base STS-B -1 target
```

其中四个参数必须传入，含义分别如下：
```
- model_type: 模型大小，必须是['base', 'large', 'base-nli', 'large-nli']之一；
- task_name: 评测数据集，必须是['STS-B', 'STS-12', 'STS-13', 'STS-14', 'STS-15', 'STS-16', 'SICK-R']之一；
- n_components: 保留的维度，如果是0，则不进行whitening，如果是负数，则保留全部维度，如果是正数，则按照所给的维度保留；
- normalized_by: 白化所用的数据集，必须是['target', 'nli']之一。
```

## 微调

自行使用Sentence-BERT（SBERT）的方式用NLI数据微调：
```
python nli_train.py base
python nli_train.py large
```

## 下载

Google官方的两个BERT模型：
- [uncased_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)
- [uncased_L-24_H-1024_A-16.zip](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip)

关于语义相似度数据集，可以参考[BERT-flow](https://github.com/bohanli/BERT-flow)的下载方式，也可以从作者提供的百度云链接下载。
- 链接: https://pan.baidu.com/s/1UfPZc7n1cPZlakiIQF8eBQ 提取码: q2tc

其中senteval目录是评测数据集汇总，senteval.zip是senteval目录的打包，两者下其一就好，weights目录包含作者事先微调好的SBERT模型和计算好的均值方差，可以复现论文中的结果，读者也可以不下载weights，评测脚本会自动生成均值方差，`nli_train.py`则可以自己训练SBERT
