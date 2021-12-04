#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 11:13
# @Author  : 刘鑫
# @FileName: sequnce_labeling_train.py
# @Software: PyCharm

# 假定已经带着正确的分类label去训练实体识别
import sys

import tensorflow as tf
import os, math
from seqeval.metrics import accuracy_score, f1_score
from bert import modeling
from bert import optimization
from tensorflow.contrib.layers.python.layers import initializers
from algorithm.kg_qa.config import Properties, LstmCRFConfig as config
from algorithm.kg_qa.NER_BERT_LSTM_CRF.lstm_crf_layer import BLSTM_CRF
from utils.IdAndLabel import id2label
from utils.EvalReport import report


def load_bert_config(path):
    """
    bert 模型配置文件
    """
    return modeling.BertConfig.from_json_file(path)


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, keep_prob, num_labels,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope='bert'
    )
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value

    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=config.lstm_size, cell_type=config.cell,
                          num_layers=config.num_layers,
                          dropout_rate=config.droupout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    (total_loss, logits, trans, pred_ids) = blstm_crf.add_blstm_crf_layer(crf_only=False)

    return (total_loss, logits, trans, pred_ids)


def get_input_data(input_file, seq_length, batch_size, is_training=True):
    def parser(record):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        }

        example = tf.parse_single_example(record, features=name_to_features)
        input_ids = example["input_ids"]
        input_mask = example["input_mask"]
        segment_ids = example["segment_ids"]
        labels = example["label_ids"]
        return input_ids, input_mask, segment_ids, labels

    dataset = tf.data.TFRecordDataset(input_file)
    # 数据类别集中，需要较大的buffer_size，才能有效打乱，或者再 数据处理的过程中进行打乱
    if is_training:
        dataset = dataset.map(parser).batch(batch_size).shuffle(buffer_size=2000)
    else:
        dataset = dataset.map(parser).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    input_ids, input_mask, segment_ids, labels = iterator.get_next()
    return input_ids, input_mask, segment_ids, labels


def id2label_f(id_list):
    seq_id2label = id2label(config.label_list)
    predictions = []
    for id in id_list:
        predictions.append(seq_id2label[id])
    return predictions


def pre2out(predicts):
    '''
    可以考虑移动到utils里面
    :param predicts:
    :return:
    '''
    outs = []
    for tmp in predicts:
        tmp_out = id2label_f(tmp.tolist())
        outs.append(tmp_out)
    return outs


def main():
    print("print start load the params...")

    tf.gfile.MakeDirs(config.model_out)

    # 配置超参数
    train_examples_len = config.train_examples_len
    valid_examples_len = config.valid_examples_len
    learning_rate = config.learning_rate
    eval_per_step = config.eval_per_step
    num_labels = config.num_labels

    num_train_steps = math.ceil(train_examples_len / config.train_batch_size)
    num_valid_steps = math.ceil(valid_examples_len / config.valid_batch_size)
    num_warmup_steps = math.ceil(num_train_steps * config.num_train_epochs * config.warmup_proportion)
    print("num_train_steps:{},  num_valid_steps:{},  num_warmup_steps:{}".format(num_train_steps, num_valid_steps,
                                                                                 num_warmup_steps))

    use_one_hot_embeddings = False
    is_training = True
    seq_len = config.max_seq_length

    init_checkpoint = Properties.init_checkpoint
    print("print start compile the bert model...")

    # 定义输入输出
    input_ids = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_ids')
    input_mask = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_mask')
    segment_ids = tf.placeholder(tf.int64, shape=[None, seq_len], name='segment_ids')
    token_labels = tf.placeholder(tf.int64, shape=[None, seq_len], name='token_labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # , name='is_training'
    bert_config_ = load_bert_config(Properties.bert_config)
    # total_loss, logits, trans, pred_ids
    (total_loss, logits, trans, pred_ids) = create_model(bert_config_, is_training, input_ids,
                                                         input_mask, segment_ids, token_labels,
                                                         keep_prob,
                                                         num_labels, use_one_hot_embeddings)

    train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps * config.num_train_epochs,
                                             num_warmup_steps, False)

    print("print start train the bert model...")

    batch_size = config.train_batch_size
    valid_batch_size = config.valid_batch_size

    init_global = tf.global_variables_initializer()

    saver = tf.train.Saver([v for v in tf.global_variables() if 'adam_v' not in v.name and 'adam_m' not in v.name],
                           max_to_keep=3)  # 保存最后top3模型

    with tf.Session() as sess:

        print("start load the pre train model")

        if init_checkpoint:
            tvars = tf.trainable_variables()
            print("trainable_variables", len(tvars))
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            print("initialized_variable_names:", len(initialized_variable_names))

            saver_ = tf.train.Saver([v for v in tvars if v.name in initialized_variable_names])
            saver_.restore(sess, init_checkpoint)
            sess.run(init_global)
        else:
            sess.run(tf.global_variables_initializer())
        print("********* train start *********")

        def train_step(ids, mask, segment, y, step, train_out_f):
            feed = {input_ids: ids,
                    input_mask: mask,
                    segment_ids: segment,
                    token_labels: y,
                    keep_prob: 0.9}
            _, out_loss, pred_ids_ = sess.run([train_op, total_loss, pred_ids], feed_dict=feed)
            acc = accuracy_score(pre2out(y), pre2out(pred_ids_))
            f1 = f1_score(pre2out(y), pre2out(pred_ids_))
            print("step :{},loss :{}, acc :{}, f1 :{}".format(step, out_loss, acc, f1))
            train_out_f.write("step :{}, loss :{},  acc :{} , f1 :{} \n".format(step, out_loss, acc, f1))
            return out_loss, pred_ids_, y

        def valid_step(ids, mask, segment, y):
            # 验证训练效果
            feed = {input_ids: ids,
                    input_mask: mask,
                    segment_ids: segment,
                    token_labels: y,
                    keep_prob: 1.0
                    }

            out_loss, p_ = sess.run([total_loss, pred_ids], feed_dict=feed)
            # f1 = f1_score(pre2out(y), pre2out(p_))
            # acc = accuracy_score(pre2out(y), pre2out(p_))
            # print("loss :{}, acc :{}, f1:{}".format(out_loss, acc, f1)) 没有必要一个批次算一次
            return out_loss, p_, y

        min_total_loss_dev = 999999
        step = 0
        if not os.path.exists(config.training_log):
            os.makedirs(config.training_log)
        for epoch in range(config.num_train_epochs):
            _ = "{:*^100s}".format(("epoch-" + str(epoch)).center(20))
            print(_)
            # 读取训练数据
            total_loss_train = 0

            input_ids2, input_mask2, segment_ids2, labels2 = get_input_data(config.train_data, seq_len, batch_size)
            train_out_f = open(os.path.join(config.training_log, "epoch-" + str(epoch) + ".txt"), 'w', encoding='utf-8')

            for i in range(num_train_steps):
                step += 1
                ids_train, mask_train, segment_train, y_train = sess.run(
                    [input_ids2, input_mask2, segment_ids2, labels2])
                out_loss, pre, y = train_step(ids_train, mask_train, segment_train, y_train, step, train_out_f)
                total_loss_train += out_loss

                if step % eval_per_step == 0 and step >= config.eval_start_step:
                    total_loss_dev = 0
                    dev_input_ids2, dev_input_mask2, dev_segment_ids2, dev_labels2 = get_input_data(config.valid_data,
                                                                                                    seq_len,
                                                                                                    valid_batch_size,
                                                                                                    False)
                    pre_list = []
                    y_list = []
                    for j in range(num_valid_steps):  # 一个 epoch 的 轮数
                        ids_dev, mask_dev, segment_dev, y_dev = sess.run(
                            [dev_input_ids2, dev_input_mask2, dev_segment_ids2, dev_labels2])
                        out_loss, pre, y = valid_step(ids_dev, mask_dev, segment_dev, y_dev)
                        total_loss_dev += out_loss
                        pre_list.extend(pre)
                        y_list.extend(y)
                    print("total_loss_dev:{}".format(total_loss_dev))
                    report(pre2out(y_list), pre2out(pre_list))

                    if total_loss_dev < min_total_loss_dev:
                        print("save model:\t%f\t>%f" % (min_total_loss_dev, total_loss_dev))
                        min_total_loss_dev = total_loss_dev
                        saver.save(sess, config.model_out + 'bert.ckpt', global_step=step)
                elif step < config.eval_start_step and step % config.auto_save == 0:
                    saver.save(sess, config.model_out + 'bert.ckpt', global_step=step)
            train_out_f.close()
            _ = "{:*^100s}".format(("epoch-" + str(epoch) + " report:").center(20))
            print("total_loss_train:{}".format(total_loss_train))
            # print(classification_report(total_true_train, total_pre_train, digits=4))


if __name__ == "__main__":
    print("********* ner start *********")
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
