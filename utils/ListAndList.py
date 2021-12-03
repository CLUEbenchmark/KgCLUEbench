#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/29 22:54
# @Author  : 刘鑫
# @FileName: ListAndList.py
# @Software: PyCharm
import random


def _index_q_list_in_k_list(q_list, k_list):
    """Known q_list in k_list, find index(first time) of q_list in k_list"""
    q_list_length = len(q_list)
    k_list_length = len(k_list)
    for idx in range(k_list_length - q_list_length + 1):
        t = [q == k for q, k in zip(q_list, k_list[idx: idx + q_list_length])]
        if all(t):
            idx_start = idx
            return idx_start


