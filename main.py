from utils import Reader, FeatureGeneration
import torch
import numpy as np
import random
from kfold import cross_validation
from SeqModel import TextRNN

class Config:
    def __init__(self):
        #地址
        self.model_type = ''
        self.data_type = ''
        self.model_saved_path = 'model/' + self.model_type + '/' +self.data_type + '.bin'
        #模型
        self.hidden_size = 128
        self.max_length = 16
        #训练
        self.seed = 520
        self.batch_size = 128
        self.val_size = 0.2
        self.train_numples = 900000
        self.num_train_epochs = 100
        self.pattern = 'cross_validation'
        self.model_saved = False
        self.require_grad = False          #词向量是否梯度反传
        self.early_stop = False
        self.require_improvement = 1000
        self.warmup_proportion = 0.1
        #设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_id = 0
        #输入
        self.out_prob = False
        self.test = False             #sh'c

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return None

def main(config):
    random_seed(config.seed)
    reader = Reader()
    train_data, test_data = reader.read_data()
    generation = FeatureGeneration()
    train_data, test_data = train_data[:1000], test_data[:1000]
    train_data, test_data = generation.convert_example_feature(train_data, config.max_length),  \
                            generation.convert_example_feature(test_data, config.max_length)
    embedding_matrix = generation.load_embedding('creative_id')
    model = TextRNN(config, embedding_matrix)
    model_trained, dev_evalution, predict_label = cross_validation(config, model, train_data, test_data)
    # if config.pattern == 'cross_validation':
    #     result_save(predic_label)




if __name__ == "__main__":
    config = Config()
    main(config)
