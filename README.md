[toc]

# KgClue_Bench

尽最大能力解耦代码，为NLP新手提供(BERT)学习平台

## 目录结构

├─algorithm # 算法 <br>
│ └─kg_qa # 算法开发示例<br>
│ │ config.py<br>
│ ├─KG 每个模块对应一个package<br>
│ │ │ es.py<br>
│ │ │ KgAnswer.py<br>
│ │ │ KgEval.py<br>
│ ├─NER<br>
│ │ │ DataMaking.py<br>
│ │ │ EntityExtract.py<br>
│ │ │ Eval.py<br>
│ │ │ Predict.py<br>
│ │ │ TrainAndValid.py<br>
├─bert 谷歌官方Bert代码存放<br>
│ │ .gitignore<br>
├─pretraining_model # 存放bert的预训练模型<br>
│ ├─chinese_rbt3_L-3_H-768_A-12 #存放示例<br>
├─raw_data # 数据集推荐添加方式,直接解压<br>
│ ├─kgClue # kg_qa项目中适配的数据集<br>
│ │ │ eval.json<br>
│ │ │ kgClue.yaml<br>
│ │ │ test_public.json<br>
│ │ │ train.json<br>
│ │ └─knowledge # 知识库<br>
│ │ Knowledge.txt<br>
└─utils<br>


## 算法排行

### **kg_qa任务** 以kgClue为训练数据集，旨在回答知识库中的问题

> #### 不同算法结构性能比较(以roberta为预训练模型)
> 这里的评估是以问题回答准确度作为标准
> 
>Model   | F1     | EM  |
>:----:| :----:  |:----:  |
>bert-crf |  66.1       |  66.0    |
>bert-lstm-crf |  63.9       |  63.6    |

#### 不同预训练模型性能比较(不代表每个模型的最佳性能)
NER (bert+crf) seq_lan=32 epoch=5

| pretraining_model      | batch | micro-f1| macro-f1| f1(##WordPiece) |f1(B-NP/I-NP)|
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| chinese_rbt3_L-3_H-768_A-12      | 40       | 93.1| 88.0 | 61.0 | 79.0 |
| chinese_rbt4_L-4_H-768_A-12   | 40        | 92.0 | 87.0 | 62.0 | 75.0 |
| chinese_rbt6_L-6_H-768_A-12   | 40        | 93.0 | 88.0 | 61.0 | 77.0 |
| chinese_rbtl3_L-3_H-1024_A-16   | 40       | 93.0 | 89.0 | 66.0 | 77.0 | 
| chinese_wwm_ext_L-12_H-768_A-12   | 40       | 93.0 | 88.0 | 63.0 | 77.0 | 

SIM (bert) seq_lan=64 epoch=5

| pretraining_model      | batch | accuracy| precision| recall |macro-f1|
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| chinese_rbt3_L-3_H-768_A-12      | 40       | 86.0| 44.3 | 2.0 | 49.0 |
| chinese_rbt4_L-4_H-768_A-12   | 40        | 93.5 | 78.3 | 73.1 | 85.9 |
| chinese_rbt6_L-6_H-768_A-12   | 40        | 93.8 | 79.2 | 74.9 | 86.7 |
| chinese_rbtl3_L-3_H-1024_A-16   | 40       |96.5 |86.4| 89.1| 92.9 | 
| chinese_wwm_ext_L-12_H-768_A-12   | 40       | 95.5| 82.1| 86.6 | 90.9 | 

## 使用示例

### 以 **kg_qa** 算法为例

> 该项目下有三个文件夹KG\NER\SIM

#### NER

1. 执行DataMaking.py **注意**: 1. 文件路径 2.脚本work路径,应该以整个KgCLUEBench为项目根目录运行
2. 执行TrainAndValid.py **注意** :训练之前设置好kg_qa目录下的config配置,其他注意点同上
3. 执行Precit.py 验证是否正常运行
4. 执行Eval.py 得出模型的评估结果,可以在训练时间断Eval模型,查看训练效果
5. 执行EntityExtract.py 将序列标注结果(Predict结果)转化为句子中的实体

#### SIM KG同理

## algorithm 贡献方法

> 在此目录下直接命名一个新的python包包含init和config文件
> 不同算法可能有多个stage，不同stage建议使用独立的python包，多个stage共享一个config

## UPDATE

******* 2021-12-3,新项目开荒

## 有问题联系1194370384@qq.com

