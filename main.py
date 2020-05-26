import torch
import random
import os
import numpy as np
from SeqModel import TextRNN
from train_eval import model_save
from kfold import cross_validation
from utils import Reader, FeatureGeneration, result_save
import logging
import time
import torch

class Config:
    def __init__(self):
        #地址
        self.model_type = 'TextRNN'
        self.data_type = 'gender'
        self.model_saved_path = 'model/' + self.model_type + '/' +self.data_type + '/'
        self.result_path = 'result/' + self.data_type + '/'
        self.logging_file = 'logging/' + self.model_type + '/' + self.data_type + '/'
        #模型
        self.hidden_size = 256
        self.max_length = 90                                                               #很敏感！！！！！！！！！
        #训练
        self.seed = 520
        self.learning_rate = 3e-3
        self.batch_size = 1024
        self.val_size = 0.2                                                                #验证集占比
        self.train_numples = 900000
        self.num_train_epochs = 10                                                          #训练轮数
        self.pattern = 'cross_validation'
        self.require_grad = False                                                           #训练过程是否更新词向量
        self.early_stop = False                                                            #验证集正确率不上升，就停止训练
        self.require_improvement = 1000
        self.warmup_proportion = 0.1                                                       #warmup轮数占比
        #设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_id = 1
        #torch.cuda.set_device(self.device_id)
        #输出
        self.out_prob = False                                                              #输出是否为概率
        self.test = True                                                                   #输出测试集预测
        self.model_saved = True                                                            #是否保存模型
        self.logging2file = True

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return None

def main(config):
    random_seed(config.seed)
    reader = Reader()    #读取数据
    train_data, test_data = reader.read_data()
    generation = FeatureGeneration()    #生成特征
    train_data, test_data = train_data[:1000], test_data[:1000]
    train_data, test_data = generation.convert_example_feature(train_data, config.max_length),  \
                            generation.convert_example_feature(test_data, config.max_length)
    embedding_matrix = generation.load_embedding('creative_id')
    model = TextRNN(config, embedding_matrix)    #定义模型
    logging.info(model)
    model_trained, predict_label = cross_validation(config, model, train_data, test_data)  #使用不同验证方式
    if config.test:  #存储预测结果
        result_save(config, examples=test_data, genders=predict_label)
    if config.model_saved:  #存储模型
        model_save(config, model=model_trained)



if __name__ == "__main__":
    config = Config()
    if config.logging2file == True:  #是否生成日志
        if not os.path.exists(config.logging_file):
            os.makedirs(config.logging_file)
        file = time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
        path = os.path.join(config.logging_file, file)
        logging.basicConfig(filename=path, format='%(levelname)s: %(message)s', level=logging.INFO)
    main(config)
