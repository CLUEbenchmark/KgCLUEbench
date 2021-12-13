[toc]

# KgClue_Bench

尽最大能力解耦代码，为NLP新手提供(BERT)学习平台

## 目录结构

├─algorithm # 算法 <br>
│ └─kg_qa # 算法开发示例<br>
│ │ config.py<br>
│ ├─KG 每个模块对应一个package<br>
│ │ │ es.py # 将知识库导入es的脚本<br>
│ │ │ KgAnswer.py # 回答问题类<br>
│ │ │ KgEval.py# 回答问题的准确度评估方法<br> 
│ │ │ KgPredict.py# 针对test.json文件生成预测结果，手动压缩之后可以提交到官网进行评估<br> 
│ ├─NER<br>
│ │ │ DataMaking.py# NER训练数据集的制作脚本<br> 
│ │ │ EntityExtract.py# 将序列标注标签转化为实体<br>
│ │ │ Eval.py# 评估代码（输出f1）<br> 
│ │ │ Predict.py# 预测类<br> 
│ │ │ TrainAndValid.py# 训练代码<br>
├─bert 谷歌官方Bert代码存放<br>
│ │ .gitignore<br>
├─pretraining_model # 存放bert的预训练模型<br>
│ ├─chinese_rbt3_L-3_H-768_A-12 #存放示例<br>
├─raw_data # 数据集推荐添加方式,直接解压<br>
│ ├─kgClue # kg_qa项目中适配的数据集<br>
│ │ │ xxx.json<br>
│ │ └─knowledge # 知识库<br>
│ │ Knowledge.txt<br>
└─utils<br>


## 算法排行

### **kg_qa任务** 以kgClue为训练数据集，旨在回答知识库中的问题

> #### 不同算法结构性能比较(以chinese_rbtl3_L-3_H-1024_A-16为预训练模型)
> 这里的评估是以问题回答准确度作为标准
> 
>Model   | F1     | EM  |
>:----:| :----:  |:----:  |
>bert-crf |  70.7      |  70.7   |
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

1. python DataMaking.py **注意**: 1. 文件路径 2.脚本work路径,应该以整个KgCLUEBench为项目根目录运行
2. python TrainAndValid.py **注意** :训练之前设置好kg_qa目录下的config配置,其他注意点同上
3. python Precit.py 验证是否正常运行
4. python Eval.py 得出模型的评估结果,可以在训练时间段Eval模型,查看训练效果
5. python EntityExtract.py 将序列标注结果(Predict结果)转化为句子中的实体

#### SIM 同理

1. python DataMaking.py **注意**: 1. 文件路径 2.脚本work路径,应该以整个KgCLUEBench为项目根目录运行
2. python TrainAndValid.py **注意** :训练之前设置好kg_qa目录下的config配置,其他注意点同上
3. python Precit.py 验证是否正常运行
4. python Eval.py 得出模型的评估结果,可以在训练时间段Eval模型,查看训练效果

#### KG
1. es.py是将知识库（这里是Knowledge.txt）导入es系统的脚本文件，只需要执行一次
2. KgAnswer.py是回答问题的类，只需要输入一个句子，即可给出结果
3. KgEval是评估问题回答能力的代码，修改文件路径即可使用
4. KgPredict是回答test.json的代码，运行完成可以生成kgclue_predict.txt，用户压缩成zip文件之后可以直接提交至clue官网。

## algorithm 贡献方法

> 在此目录下直接命名一个新的python包包含init和config文件
> 不同算法可能有多个stage，不同stage建议使用独立的python包，多个stage共享一个config

## UPDATE

******* 2021-12-3,新项目开荒
******* 2021-12-12,完整流程测试通过

## 有问题联系1194370384@qq.com

