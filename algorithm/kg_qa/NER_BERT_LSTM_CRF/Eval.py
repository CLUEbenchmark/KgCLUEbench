#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/20 10:54
# @Author  : 刘鑫
# @FileName: seq_eval.py
# @Software: PyCharm
import json
import os

from algorithm.kg_qa.NER_BERT_LSTM_CRF.Predict import Predict
from algorithm.kg_qa.NER.DataMaking import DataMaking
from algorithm.kg_qa.config import LstmCRFConfig as config
from utils.EvalReport import report


class Eval(object):
    def __init__(self, MODEL_PATH):
        self.data_make = DataMaking(do_lower_case=True, max_seq_length=config.max_seq_length)
        self.predict = Predict(MODEL_PATH)
        self.seq_id2label = self.predict.seq_id2label

    def id2label_f(self, id_list):
        predictions = []
        for id in id_list:
            predictions.append(self.seq_id2label[id])
        return predictions

    def do_eval(self, data_files=["../raw_data/test.json"]):

        for data_file in data_files:
            y_true = []
            y_pred = []
            with open(data_file, 'r', encoding='utf-8') as f:
                while True:
                    line = f.readline()
                    if line:
                        line = json.loads(line)
                        text = line["question"]
                        answer = line["answer"]
                        entity = answer.split("|||")[0].split("（")[0]

                        token_label = self.data_make.text2label(text, entity)
                        feature = self.data_make.convert_single_example(text, token_label)
                        label = self.id2label_f(feature.label_ids)
                        predict_label = self.predict.predict_one(text, TEST_MODE=True)

                        y_true.append(label)
                        y_pred.append(predict_label)

                    else:
                        break
            report(y_true, y_pred)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    MODEL_PATH = config.model_out
    ner_eval = Eval(MODEL_PATH)
    ner_eval.do_eval(data_files=["raw_data/kgClue/test_public.json"])
