#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 14:33
# @Author  : 刘鑫
# @FileName: ner_data_making.py
# @Software: PyCharm
import collections
import json
import os
import tensorflow as tf
from elasticsearch import Elasticsearch
import random

random.seed(1024)

from bert import tokenization
from algorithm.kg_qa.config import Properties, SimConfig as config


def getAttribute(Attributes, attribute):
    index = len(Attributes) - 2
    flag = False
    if attribute in Attributes:
        Attributes.remove(attribute)
        flag = True

    x = random.randint(0, index)
    out = Attributes[x]

    if flag:
        Attributes.append(attribute)

    return out


class DataMaking(object):
    def __init__(self, fake_example_nums=5, es_host="127.0.0.1", es_port="9200", do_lower_case=True,
                 max_seq_length=128):
        self.task_name = "SIM"
        self.bert_tokenizer = tokenization.FullTokenizer(vocab_file=Properties.vocab_file,
                                                         do_lower_case=do_lower_case)  # 初始化 bert_token 工具
        self.fake_example_nums = fake_example_nums
        self.max_seq_length = max_seq_length

        self.es = Elasticsearch([":".join((es_host, es_port))])
        # self.Attributes = list(set(self.getAttribute())) # 更换数据集时重新运行打印出来之后放到config里面

    def makdir(self, OUTPUT_DIR, DIR_NAME):
        if not os.path.exists(os.path.join(os.path.dirname(__file__), OUTPUT_DIR)):
            os.makedirs(os.path.join(os.path.dirname(__file__), OUTPUT_DIR))
        if not os.path.exists(os.path.join(os.path.dirname(__file__), OUTPUT_DIR, DIR_NAME)):
            os.makedirs(os.path.join(os.path.dirname(__file__), OUTPUT_DIR, DIR_NAME))

    def convert_single_example(self, text, attribute, label=0, test_mode=False):

        def _truncate_seq_pair(tokens_a, tokens_b, max_length):
            """Truncates a sequence pair in place to the maximum length."""
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

        text_token = self.bert_tokenizer.tokenize(text)
        attribute_tokens = self.bert_tokenizer.tokenize(attribute)
        # print("att", len(attribute), attribute) 调试
        tokens_b = self.bert_tokenizer.convert_tokens_to_ids(attribute_tokens) * (len(text_token) // len(attribute))
        if int(label) == 1:
            label_ids = [0, 1]
        else:
            label_ids = [1, 0]

        _truncate_seq_pair(text_token, tokens_b, self.max_seq_length - 3)  # 很重要
        #
        tokens = []
        segment_ids = []
        # 添加起始位置
        tokens.append("[CLS]")
        segment_ids.append(0)

        for token in text_token:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)

        for token in tokens_b:
            input_ids.append(token)
            segment_ids.append(1)

        input_ids.append(self.bert_tokenizer.convert_tokens_to_ids(["[SEP]"])[0])  # 102
        segment_ids.append(1)
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            tokens.append("[Padding]")

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        if test_mode:
            feature = (input_ids, input_mask, segment_ids)
        else:
            feature = config.InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                is_real_example=True)
        return feature

    def write2txt(self, text, attribute, label, token_label_f, tf_writer):
        # 写入txt
        example = text + "\t" + str(attribute) + "\t" + str(label)
        token_label_f.write(example + "\n")

        # 写人tf_record
        feature = self.convert_single_example(text, attribute, label)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        tf_writer.write(tf_example.SerializeToString())

    def input2output(self, INPUT_DATA_PATHS, OUTPUT_DIR, is_init_attribute=False):
        if is_init_attribute:
            Attributes = []
            for data_path in INPUT_DATA_PATHS:
                with open(data_path, "r", encoding='utf-8') as f:
                    for line in f.readlines():
                        line = json.loads(line)
                        attribute = line["answer"].split("|||")[1].strip()
                        Attributes.append(attribute)
            Attributes = list(set(Attributes))
        else:
            Attributes = config.Attributes

        for data_path in INPUT_DATA_PATHS:
            file_name = data_path.split('/')[-1]
            file_type = file_name.split('.')[0]
            self.makdir(OUTPUT_DIR, file_type)  # 生成文件名对应的文件夹
            OUT_PATH = os.path.join(os.path.dirname(__file__), OUTPUT_DIR, file_type)
            # 创建需要写入的文件
            token_label_f = open(os.path.join(OUT_PATH, "token_label.txt"), "w", encoding='utf-8')
            tf_writer = tf.python_io.TFRecordWriter(os.path.join(OUT_PATH, file_type + ".tf_record"))
            with open(data_path, "r", encoding='utf-8') as f:
                count_numbers = 0
                while True:
                    line = f.readline()
                    if line:
                        line = json.loads(line)
                        text = line["question"]
                        answer = line["answer"]
                        attribute = answer.split("|||")[1].strip()
                        entity = answer.split("|||")[0].split("（")[0]
                        self.write2txt(text, attribute, 1, token_label_f, tf_writer)

                        # 使用es获取相关属性的负类
                        body = {
                            "query": {
                                "term": {
                                    "entity.keyword": entity
                                }
                            }
                        }
                        es_results = self.es.search(index="kbqa-data", doc_type="kbList", body=body,
                                                    size=self.fake_example_nums)

                        attribute_list = list()
                        for i in range(len(es_results['hits']['hits'])):
                            relation = es_results['hits']['hits'][i]['_source']['relation']
                            attribute_list.append(relation)

                        # 去重复
                        attribute_list = list(set(attribute_list))
                        for fake_attribute in attribute_list:
                            # 1代表true
                            # 0代表true
                            if len(fake_attribute) == 0:
                                continue
                            self.write2txt(text, fake_attribute, 0, token_label_f, tf_writer)

                        # 随机获得不相关负类
                        for i in range(self.fake_example_nums):
                            # 1代表true
                            # 0代表true
                            fake_attribute = getAttribute(Attributes, attribute)
                            if len(fake_attribute) == 0:
                                continue
                            self.write2txt(text, fake_attribute, 0, token_label_f, tf_writer)

                        count_numbers += 1
                        if count_numbers % 10000 == 0:
                            print("Writing example %d " % (count_numbers))
                    else:
                        break
            print("all numbers", count_numbers)
            token_label_f.close()
            tf_writer.close()


if __name__ == '__main__':
    INPUT_DATA_PATHS = ["raw_data/kgClue/train.json", "raw_data/kgClue/eval.json", "raw_data/kgClue/test_public.json"]
    OUTPUT_DIR = "./data"
    fake_example_nums = 2 # 会影响sim模型数据集的个数需要及时更新config里面的值
    es_host = "127.0.0.1"
    es_port = "9200"
    sim_data_make = DataMaking(fake_example_nums, es_host, es_port, do_lower_case=True,
                               max_seq_length=config.max_seq_length)
    is_init_attribute = False
    sim_data_make.input2output(INPUT_DATA_PATHS, OUTPUT_DIR, is_init_attribute=is_init_attribute)
