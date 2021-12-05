[toc]

# KgClue_Bench

尽最大能力解耦代码，为NLP新手提供(BERT)学习平台

## 目录结构

├─algorithm # 算法 <br>
│ └─kg_qa # 算法开发示例<br>
│ │ config.py<br>
│ │  __init__.py<br>
│ │<br>
│ ├─KG 每个模块对应一个package<br>
│ │ │ es.py<br>
│ │ │ KgAnswer.py<br>
│ │ │ KgEval.py<br>
│ │ │  __init__.py<br>
│ │<br>
│ ├─NER<br>
│ │ │ DataMaking.py<br>
│ │ │ EntityExtract.py<br>
│ │ │ Eval.py<br>
│ │ │ Predict.py<br>
│ │ │ TrainAndValid.py<br>
│ │ │  __init__.py<br>
│ │<br>
│ ├─SIM<br>
│ │ │ DataMaking.py<br>
│ │ │ Eval.py<br>
│ │ │ Predict.py<br>
│ │ │ TrainAndValid.py<br>
│ │ │  __init__.py<br>
│<br>
├─bert 谷歌官方Bert代码存放<br>
│ │ .gitignore<br>
│ │  __init__.py<br>
│<br>
├─pretraining_model # 存放bert的预训练模型<br>
│ │ readme.txt<br>
│ │<br>
│ ├─chinese_rbt3_L-3_H-768_A-12 #存放示例<br>
│ │ bert_config.json<br>
│ │ bert_model.ckpt.data-00000-of-00001<br>
│ │ bert_model.ckpt.index<br>
│ │ bert_model.ckpt.meta<br>
│ │ vocab.txt<br>
│<br>
├─raw_data # 数据集推荐添加方式,直接解压<br>
│ ├─kgClue # kg_qa项目中适配的数据集<br>
│ │ │ eval.json<br>
│ │ │ kgClue.yaml<br>
│ │ │ test_public.json<br>
│ │ │ train.json<br>
│ │ │<br>
│ │ └─knowledge # 知识库<br>
│ │ Knowledge.txt<br>
│ │ README.md<br>
│<br>
└─utils<br>
│ DrawTrain.py<br>
│ EvalReport.py<br>
│ IdAndLabel.py<br>
│ ListAndList.py<br>
│  __init__.py<br>

## 算法排行

### kg_qa 以kgClue为训练数据集，旨在回答知识库中的问题

### 不同算法结构性能比较(以roberta—large为预训练模型)

| Model   | F1     | EM  |
| :----:| :----:  |:----:  |
| bert-crf |  66.1       |  66.0    |
| bert-lstm-crf |  63.9       |  63.6    |

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

