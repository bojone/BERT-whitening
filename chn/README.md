# BERT-whitening 中文测试

BERT-whitening在常见中文数据集上的测试，包含[ATEC](https://github.com/IceFlameWorm/NLP_Datasets/tree/master/ATEC)、[BQ](http://icrc.hitsz.edu.cn/info/1037/1162.htm)、[LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html)、[PAWSX](https://arxiv.org/abs/1908.11828)、[STS-B](https://github.com/pluto-junzeng/CNSD)共5个任务。

## 文件

```
- utils.py  工具函数
- eval.py  评测主文件
```

## 评测

命令格式：
```
python eval.py [model_type] [pooling] [task_name] [n_components]
```

使用例子：
```
python eval.py BERT cls ATEC 256
```

其中四个参数必须传入，含义分别如下：
```
- model_type: 模型，必须是['BERT', 'RoBERTa', 'WoBERT', 'RoFormer', 'BERT-large', 'RoBERTa-large', 'SimBERT', 'SimBERT-tiny', 'SimBERT-small']之一；
- pooling: 池化方式，必须是['first-last-avg', 'last-avg', 'cls', 'pooler']之一；
- task_name: 评测数据集，必须是['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']之一；
- n_components: 保留的维度，如果是0，则不进行whitening，如果是负数，则保留全部维度，如果是正数，则按照所给的维度保留；
```

## 下载

Google官方的两个BERT模型：
- BERT：[chinese_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
- RoBERTa：[chinese_roberta_wwm_ext_L-12_H-768_A-12.zip](https://github.com/ymcui/Chinese-BERT-wwm)
- NEZHA：[NEZHA-base-WWM](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)
- WoBERT：[chinese_wobert_plus_L-12_H-768_A-12.zip](https://github.com/ZhuiyiTechnology/WoBERT)
- RoFormer：[chinese_roformer_L-12_H-768_A-12.zip](https://github.com/ZhuiyiTechnology/roformer)
- BERT-large：[uer/mixed_corpus_bert_large_model.zip](https://github.com/dbiir/UER-py)
- RoBERTa-large：[chinese_roberta_wwm_ext_L-12_H-768_A-12.zip](https://github.com/ymcui/Chinese-BERT-wwm)
- NEZHA-large：[NEZHA-large-WWM](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)
- SimBERT: [chinese_simbert_L-12_H-768_A-12.zip](https://github.com/ZhuiyiTechnology/simbert)
- SimBERT-small: [chinese_simbert_L-6_H-384_A-12.zip](https://github.com/ZhuiyiTechnology/simbert)
- SimBERT-tiny: [chinese_simbert_L-4_H-312_A-12.zip](https://github.com/ZhuiyiTechnology/simbert)

关于语义相似度数据集，可以从数据集对应的链接自行下载，也可以从作者提供的百度云链接下载。
- 链接: https://pan.baidu.com/s/1d6jSiU1wHQAEMWJi7JJWCQ 提取码: qkt6

其中senteval_cn目录是评测数据集汇总，senteval_cn.zip是senteval目录的打包，两者下其一就好。
