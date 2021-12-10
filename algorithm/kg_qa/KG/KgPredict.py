#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/10 13:11
# @Author  : 刘鑫
# @FileName: KgPredict.py
# @Software: PyCharm

import json
import os

from KgAnswer import KgAnswer
from algorithm.kg_qa.config import NerConfig, SimConfig

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    NER_MODEL_PATH = NerConfig.model_out
    SIM_MODEL_PATH = SimConfig.model_out
    es_host = "127.0.0.1"
    es_port = "9200"
    kg = KgAnswer(NER_MODEL_PATH, SIM_MODEL_PATH, es_host, es_port)
    out_f = open("./kgclue_predict.txt", "w", encoding='utf-8')
    with open(r"C:\Users\11943\Documents\GitHub\KgCLUEbench\raw_data\kgClue\test.json", 'r', encoding='utf-8') as f:
        count_number = 0
        while True:

            line = f.readline()
            if line:

                line = json.loads(line)
                sentence = line["question"]
                best_answer, best_attribute, entitys = kg.answer(sentence)
                tmp = dict()
                tmp["id"] = count_number
                tmp["answer"] = str(entitys) + " ||| " + str(best_attribute) + " ||| " + str(best_answer)
                x = json.dumps(tmp, ensure_ascii=False)
                out_f.write(x+"\n")
                # print(x)
                count_number += 1
                # break
                # {"id": 0, "answer": "刘质平 ||| 师从 ||| 李叔同"}

            else:
                break
    out_f.close()
