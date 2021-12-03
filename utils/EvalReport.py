#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/30 22:03
# @Author  : 刘鑫
# @FileName: EvalReport.py
# @Software: PyCharm


def report(y_true, y_pred):
    '''

    :param y_true: [[]]
    :param y_pred: [[]]
    :return:
    '''

    from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
    from seqeval.metrics import classification_report

    acc_s = accuracy_score(y_true, y_pred)
    precision_s = precision_score(y_true, y_pred)
    recall_s = recall_score(y_true, y_pred)
    f1_s = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print(f'\t\t准确率为： {acc_s}')
    print(f'\t\t查准率为： {precision_s}')
    print(f'\t\t召回率为： {recall_s}')
    print(f'\t\tf1值为： {f1_s}')
    print(report)
