#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/20 10:54
# @Author  : 刘鑫
# @FileName: Eval.py
# @Software: PyCharm
import json
import os

from sklearn.metrics import f1_score, accuracy_score,recall_score,precision_score

from algorithm.kg_qa.config import Properties, SimConfig as config
from algorithm.kg_qa.SIM.Predict import Predict


# 评估说明：既然是关注序列标注模型的分类效果，文本分类结果应该给予正确的

class Eval(object):
    def __init__(self, MODEL_PATH):
        self.sim = Predict(MODEL_PATH)

    def do_eval(self, data_files=["../raw_data/test.json"]):

        for data_file in data_files:
            y_true = []
            y_pred = []
            with open(data_file, 'r', encoding='utf-8') as f:
                while True:
                    line = f.readline()
                    if line:
                        text = line.split("\t")[0]
                        attribute = line.split("\t")[1]
                        t_label = line.split("\t")[2]

                        if int(t_label) == 1:
                            t_label = True
                        else:
                            t_label = False

                        predict_label ,_= self.sim.predict_one(text, attribute, TEST_MODE=True)

                        y_true.append(t_label)
                        y_pred.append(predict_label)

                    else:
                        break

            macro = f1_score(y_true, y_pred, average='macro', zero_division=1)
            accuracy = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)

            print(f'\t\t准确率为： {accuracy}')
            print(f'\t\tf1值为： {macro}')
            print(f'\t\trecall值为： {recall}')
            print(f'\t\tprecision值为： {precision}')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用不存在的显卡预测即用cpu
    MODEL_PATH = config.model_out
    sim_eval = Eval(MODEL_PATH)
    sim_eval.do_eval(data_files=[r"C:\Users\11943\Documents\GitHub\KgCLUEbench\algorithm\kg_qa\SIM\data\test_public\token_label.txt"])
    # y = [True,False,False,True,False]
    # p = [False,False,False,True,True]
    # macro  =f1_score(y, p, average='macro', zero_division=1)
    # accuracy = accuracy_score(y, p)
    # recall = recall_score(y, p)
    # precision = precision_score(y, p)
    # print(macro)
    # print(accuracy)
    # print(recall)
    # print(precision)
