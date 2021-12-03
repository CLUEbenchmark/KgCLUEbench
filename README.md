# KgClue_Bench

重构KgClue的项目代码，尽我最大能力解耦代码，为NLP新手提供一站式学习平台

## 项目运行方法

**所有的python文件必须以KgClue_Bench为working dir**

## 文件夹解释

bert 谷歌bert的官方代码，不做任何本地化修改，原汁原味

pretraining_model 放置已经预训练好的模型

algorithm 放置独立算法的python包

raw_data 放置原始数据集
> 以ske数据集为例，除了放置train、valid、test之外，推荐存放数据集相关信息的config（.json .txt .py .yaml)文件

