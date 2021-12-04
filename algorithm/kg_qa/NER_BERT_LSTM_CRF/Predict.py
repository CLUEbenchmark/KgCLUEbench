#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/16 10:42
# @Author  : 刘鑫
# @FileName: seq_predict.py
# @Software: PyCharm

import os
import sys

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from bert import tokenization
from algorithm.kg_qa.NER.DataMaking import DataMaking
from utils.IdAndLabel import id2label
from algorithm.kg_qa.config import Properties, LstmCRFConfig as config


# 预测类
class Predict(object):

    def __init__(self, MODEL_PATH):
        self.model_path = MODEL_PATH

        self.seq_id2label = id2label(config.label_list)

        # 准备token
        self.tokenizer_ = tokenization.FullTokenizer(vocab_file=Properties.vocab_file)
        self.data_making = DataMaking(do_lower_case=True, max_seq_length=config.max_seq_length)
        self.sess = self.load_model()

        self.input_ids = self.sess.graph.get_tensor_by_name("input_ids:0")
        self.input_mask = self.sess.graph.get_tensor_by_name("input_mask:0")
        self.segment_ids = self.sess.graph.get_tensor_by_name("segment_ids:0")
        self.keep_prob = self.sess.graph.get_tensor_by_name("keep_prob:0")
        # 预测的结果
        self.p = self.sess.graph.get_tensor_by_name("ReverseSequence_1:0")

        # x =[n.name for n in tf.get_default_graph().as_graph_def().node]
        # print(x)

    def load_model(self):
        try:
            checkpoint = tf.train.get_checkpoint_state(self.model_path)
            input_checkpoint = checkpoint.model_checkpoint_path
            print("[INFO] input_checkpoint:", input_checkpoint)
        except Exception as e:
            input_checkpoint = self.model_path
            print("[INFO] Model folder", self.model_path, repr(e))

        # We clear devices to allow TensorFlow to control on which device it will load operations
        clear_devices = True
        tf.reset_default_graph()
        # We import the meta graph and retrieve a Saver
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We start a session and restore the graph weights
        sess_ = tf.Session()
        saver.restore(sess_, input_checkpoint)

        return sess_

    def predict_one(self, PREDICT_TXT, TEST_MODE=False):

        feature = self.data_making.convert_single_example(PREDICT_TXT, TEST_MODE=TEST_MODE)

        def id2label_f(id_list):
            predictions = []
            if TEST_MODE:
                for id in id_list:
                    predictions.append(self.seq_id2label[id])
            else:
                for id in id_list:
                    predictions.append(self.seq_id2label[id])
                    if id == 4:
                        break
            return predictions

        feed = {self.input_ids: [feature[0]],
                self.input_mask: [feature[1]],
                self.segment_ids: [feature[2]],
                self.keep_prob: 1.0}

        probs = self.sess.run(self.p, feed)

        out = id2label_f(probs[0])

        if TEST_MODE:
            return out
        else:
            return out[1:-1]


if __name__ == '__main__':
    MODEL_PATH = config.model_out
    PREDICT_TXT = "5·13土耳其索玛矿难的坐标是什么？"

    ner = Predict(MODEL_PATH)
    print("test predict: ", ner.predict_one(PREDICT_TXT, TEST_MODE=True))
