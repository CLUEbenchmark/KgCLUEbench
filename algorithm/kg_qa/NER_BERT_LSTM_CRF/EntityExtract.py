#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/27 14:56
# @Author  : 刘鑫
# @FileName: entity_extract.py
# @Software: PyCharm
import json
import os

from bert import tokenization
from algorithm.kg_qa.config import Properties, LstmCRFConfig as config
from algorithm.kg_qa.NER_BERT_LSTM_CRF.Predict import Predict


def isEntityInText(text, entity):
    text_list = list(text)
    entity_list = list(entity)
    M = len(text_list)
    N = len(entity_list)

    i = 0
    while i <= M - N:
        j = 0
        space_nums = 0
        match = []
        while j < N:
            tt = text_list[i + j].lower()
            ee = entity_list[j].lower()
            if tt == ee:
                j += 1
                match.append(ee)
                continue
            elif j > 0 and tt == " ":
                i += 1
                space_nums += 1
            else:
                j += 1
                break
        if j == N and entity_list == match:
            return i - space_nums, space_nums
        i += 1
    return -1, -1


class EntityExtract(object):
    def __init__(self, MODEL_PATH):
        self.predict = Predict(MODEL_PATH)
        self.tokenizer_ = tokenization.FullTokenizer(vocab_file=Properties.vocab_file)

    def extract(self, sentence):
        '''
        将预测的token_label对应到句子中的字，抽取出中文实体，可能存在问题：空格缺失、大小写不一致，不过这里采用的策略是原样输出。在计算f1和em时这两点问题可以忽略
        :param sentence:
        :return: entitys
        '''

        def _merge_WordPiece_and_single_word(entity_sort_list):
            entity_sort_tuple_list = []
            for a_entity_list in entity_sort_list:
                entity_content = ""
                entity_type = None
                for idx, entity_part in enumerate(a_entity_list):
                    if idx == 0:
                        entity_type = entity_part
                        if entity_type[:2] not in ["B-", "I-"]:
                            break
                    else:
                        if entity_part.startswith("##"):
                            entity_content += entity_part.replace("##", "")
                        else:
                            entity_content += entity_part
                if entity_content != "":
                    entity_sort_tuple_list.append((entity_type[2:], entity_content))
            return entity_sort_tuple_list

        ner_out = self.predict.predict_one(sentence, TEST_MODE=True)

        def preprocessing_model_token_lable(predicate_token_label_list, token_in_list_lenth):
            if predicate_token_label_list[0] == "[CLS]":
                predicate_token_label_list = predicate_token_label_list[1:]  # y_predict.remove('[CLS]')
            if len(predicate_token_label_list) > token_in_list_lenth:  # 只取输入序列长度即可
                predicate_token_label_list = predicate_token_label_list[:token_in_list_lenth]
            return predicate_token_label_list

        predicate_token_label_list = preprocessing_model_token_lable(ner_out, len(ner_out))

        entity_sort_list = []
        entity_part_list = []

        token_in_not_UNK = self.tokenizer_.tokenize_not_UNK(sentence)
        for idx, token_label in enumerate(predicate_token_label_list):
            if token_label == "O":
                if len(entity_part_list) > 0:
                    entity_sort_list.append(entity_part_list)
                    entity_part_list = []
            if token_label.startswith("B-"):
                if len(entity_part_list) > 0:  # 适用于 B- B- *****的情况
                    entity_sort_list.append(entity_part_list)
                    entity_part_list = []
                entity_part_list.append(token_label)
                entity_part_list.append(token_in_not_UNK[idx])
                if idx == len(predicate_token_label_list) - 1:
                    entity_sort_list.append(entity_part_list)
            if token_label.startswith("I-") or token_label == "[##WordPiece]":
                if len(entity_part_list) > 0:
                    entity_part_list.append(token_in_not_UNK[idx])
                    if idx == len(predicate_token_label_list) - 1:
                        entity_sort_list.append(entity_part_list)
            if token_label == "[SEP]":
                break

        entity_sort_tuple_list = _merge_WordPiece_and_single_word(entity_sort_list)
        entitys = []
        for entity in entity_sort_tuple_list:
            if entity[0] == "NP":
                start, space_nums = isEntityInText(sentence, entity[1])
                if start != -1:
                    end = len(entity[1]) + space_nums + start
                    entitys.append(sentence[start:end])
        return entitys


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    MODEL_PATH = config.model_out
    ee = EntityExtract(MODEL_PATH)
    ff = open("./out.txt", 'w', encoding='utf-8')
    true_count=0
    with open(r"C:\Users\11943\Documents\GitHub\KgClue_Bench\raw_data\kgClue\test_public.json", "r",
              encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line:
                line = json.loads(line)
                text = line["question"]
                answer = line["answer"]
                entity = answer.split("|||")[0].split("（")[0]
                p_entity = ee.extract(text)
                print(entity,"".join(p_entity))
                if entity ==  "".join(p_entity):
                    true_count+=1
                ff.write(text + "\t" + entity + "\t" + "".join(p_entity) + "\n")
            else:
                break
    ff.close()
    print(true_count/2000)
