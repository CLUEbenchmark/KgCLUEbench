#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/30 17:45
# @Author  : 刘鑫
# @FileName: IdAndLabel.py
# @Software: PyCharm

def id2label(label_list):
    out = dict()
    for idx, label in enumerate(label_list):
        out[idx] = label
    return out
