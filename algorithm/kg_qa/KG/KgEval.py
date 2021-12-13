#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/28 21:00
# @Author  : 刘鑫
# @FileName: KgEval.py
# @Software: PyCharm
import json
import os

from KgAnswer  import KgAnswer
from algorithm.kg_qa.config import NerConfig, SimConfig

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    NER_MODEL_PATH = NerConfig.model_out
    SIM_MODEL_PATH = SimConfig.model_out
    es_host = "127.0.0.1"
    es_port = "9200"
    kg = KgAnswer(NER_MODEL_PATH, SIM_MODEL_PATH, es_host, es_port)
    em_score = 0
    f1_score = 0
    with open(r"C:\Users\11943\Documents\GitHub\KgCLUEbench\raw_data\kgClue\test_public.json", 'r', encoding='utf-8') as f:
        count_number = 0
        while True:

            line = f.readline()
            if line:
                count_number += 1
                line = json.loads(line)
                sentence = line["question"]
                answer = line["answer"].split("|||")[2].strip()
                p_answers,_,_ = kg.answer(sentence)
                # print(answer,p_answers)

                em_score += compute_exact_match(p_answers, answer)
                f1_score += compute_f1(p_answers, answer)

            else:
                break

    em = em_score/count_number
    f1 = f1_score/count_number
    print(em)
    print(f1)