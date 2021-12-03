[toc]
# KgClue_Bench

尽最大能力解耦代码，为NLP新手提供(BERT)学习平台

## 目录结构

├─algorithm # 算法
│  └─kg_qa # 算法开发示例
│      │  config.py
│      │  __init__.py
│      │
│      ├─KG 每个模块对应一个package
│      │  │  es.py
│      │  │  KgAnswer.py
│      │  │  KgEval.py
│      │  │  __init__.py
│      │
│      ├─NER 
│      │  │  DataMaking.py
│      │  │  EntityExtract.py
│      │  │  Eval.py
│      │  │  Predict.py
│      │  │  TrainAndValid.py
│      │  │  __init__.py
│      │
│      ├─SIM
│      │  │  DataMaking.py
│      │  │  Eval.py
│      │  │  Predict.py
│      │  │  TrainAndValid.py
│      │  │  __init__.py
│
├─bert 谷歌官方Bert代码存放
│  │  .gitignore
│  │  __init__.py
│
├─pretraining_model # 存放bert的预训练模型
│  │  readme.txt
│  │
│  ├─chinese_rbt3_L-3_H-768_A-12 #存放示例
│  │      bert_config.json
│  │      bert_model.ckpt.data-00000-of-00001
│  │      bert_model.ckpt.index
│  │      bert_model.ckpt.meta
│  │      vocab.txt
│
├─raw_data # 数据集推荐添加方式,直接解压
│  ├─clunner2020 # clunner2020的数据集
│  │      data_processor_seq.py
│  │      README.md
│  │      test.json
│  │      train.json
│  │      valid.json
│  │
│  ├─kgClue # kg_qa项目中适配的数据集
│  │  │  eval.json
│  │  │  kgClue.yaml
│  │  │  test_public.json
│  │  │  train.json
│  │  │
│  │  └─knowledge # 知识库
│  │          Knowledge.txt
│  │          README.md
│  └─ske # 中文实体识别数据集
│          all_50_schemas
│          ske.yaml
│          test.json
│          train.json
│          valid.json
│
└─utils
    │  DrawTrain.py
    │  EvalReport.py
    │  IdAndLabel.py
    │  ListAndList.py
    │  __init__.py

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

## 有问题联系1194370384@qq.com

