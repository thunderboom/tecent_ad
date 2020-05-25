# coding: UTF-8
import os
import copy
import logging
import numpy as np
import torch
from sklearn import metrics
import time
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
import torch.nn as nn
logger = logging.getLogger(__name__)


def model_train(config, model, train_iter, dev_iter=None):
    '''训练模型'''
    start_time = time.time()
    model = model.to(config.device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    t_total = len(train_iter) * config.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=t_total * config.warmup_proportion, num_training_steps=t_total
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Train Num examples = %d", config.train_numples*(1-config.val_size))
    logger.info("  Dev Num examples = %d", config.train_numples*config.val_size)
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Instantaneous batch size GPU/CPU = %d", config.batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Train device:%s, id:%d", config.device, config.device_id)

    global_batch = 0  # 记录进行到多少batch
    dev_best_acc = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    predict_all = []
    labels_all = []
    best_model = copy.deepcopy(model)
    for epoch in range(config.num_train_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_train_epochs))
        # scheduler.step() # 学习率衰减
        for i, (creative_id, age, gender) in enumerate(train_iter):
            global_batch += 1
            model.train()
            creative_id = torch.tensor(creative_id).type(torch.LongTensor).to(config.device)
            age = torch.tensor(age).type(torch.LongTensor).to(config.device)
            gender = torch.tensor(gender).type(torch.LongTensor).to(config.device)
            outputs, loss = model(creative_id, age, gender)
            model.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            labels = list(np.array(gender.cpu().detach().numpy(), dtype='int'))
            predic = list(np.array(outputs.cpu().detach().numpy() >= 0.50, dtype='int'))
            labels_all.extend(labels)
            predict_all.extend(predic)

            if global_batch % 10 == 0:

                train_acc = metrics.accuracy_score(labels_all, predict_all)
                predict_all = []
                labels_all = []

                # dev 数据
                dev_acc, dev_loss = train_acc, loss
                improve = ''
                if dev_iter is not None:
                    dev_acc, dev_loss = model_evaluate(config, model, dev_iter)

                    if dev_acc > dev_best_acc:
                        dev_best_acc = dev_acc
                        improve = '*'
                        last_improve = global_batch
                        best_model = copy.deepcopy(model)
                    else:
                        improve = ''

                time_dif = time.time() - start_time
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.6f},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.6f},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                logging.info(msg.format(global_batch, loss.cpu().data.item(), train_acc, dev_loss.cpu().data.item(), dev_acc, time_dif, improve))
            if config.early_stop and global_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    return best_model


def model_evaluate(config, model, val_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = []
    predict_prob = []
    labels_all = []
    with torch.no_grad():
        for i, (creative_id, age, gender) in enumerate(val_iter):
            creative_id = torch.tensor(creative_id).type(torch.LongTensor).to(config.device)
            age = torch.tensor(age).type(torch.LongTensor).to(config.device)
            gender = torch.tensor(gender).type(torch.LongTensor).to(config.device)
            outputs, loss = model(creative_id, age, gender)
            labels = list(gender.cpu().detach().numpy())
            predic = list(np.array(outputs.cpu().detach().numpy() >= 0.50, dtype='int'))
            outputs = list(outputs.cpu().detach().numpy())
            labels_all.extend(labels)
            predict_all.extend(predic)
            predict_prob.extend(outputs)
            if not test:
                loss_total += loss
    # predict_all = [label[0] for label in predict_all]
    # predict_prob = [label[0] for label in predict_prob]
    if test == True:
        if config.out_prob:
            return predict_prob
        return predict_all
    acc = metrics.accuracy_score(labels_all, predict_all)

    return acc, loss_total / len(val_iter)


#
#
# def model_test(config, model, test_iter):
#     # test!
#     logger.info("***** Running testing *****")
#     logger.info("  Test Num examples = %d", config.test_num_examples)
#     start_time = time.time()
#     test_acc, test_loss, test_report, test_confusion, _ = model_evaluate(config, model, test_iter, test=True)
#     msg = 'Test Loss: {0:>5.4},  Test Acc: {1:>6.2%}'
#     logger.info(msg.format(test_loss, test_acc))
#     logger.info("Precision, Recall and F1-Score...")
#     logger.info(test_report)
#     logger.info("Confusion Matrix...")
#     logger.info(test_confusion)
#     time_dif = time.time() - start_time
#     logger.info("Time usage:%.6fs", time_dif)


def model_save(config, model):
    if not os.path.exists(config.model_saved_path):
        os.makedirs(config.model_saved_path)
    file_name = os.path.join(config.model_saved_path, config.data_type+'.pkl')
    torch.save(model.state_dict(), file_name)
    logger.info("model saved, path: %s", file_name)

#
# def model_load(config, model, num=0 ,device='cpu'):
#     device_id = config.device_id
#     file_name = os.path.join(config.save_path[num], config.models_name[num]+'.pkl')
#     logger.info('loading model: %s', file_name)
#     model.load_state_dict(torch.load(file_name,
#                                      map_location=device if device == 'cpu' else "{}:{}".format(device, device_id)))
#
