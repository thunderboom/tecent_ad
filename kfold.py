"""根据切分形式，进行训练并验证"""
from torch.utils.data import DataLoader
from utils import AdvData
from sklearn.model_selection import train_test_split
from train_eval import model_train, model_evaluate



def cross_validation(config, model, train_data, test_data):
    if config.pattern == 'cross_validation':
        train_data, val_data = train_test_split(train_data, test_size=config.val_size, random_state=config.seed) #分训练集和验证集
        train_data, val_data, test_data = \
            AdvData(train_data, config), AdvData(val_data, config), AdvData(test_data, config)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size)
        val_loader = DataLoader(val_data, shuffle=False, batch_size=config.batch_size)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
        model_trained = model_train(config, model, train_loader, val_loader)            #训练模型
        predict_label = None
        if config.test == True:
            predict_label = model_evaluate(config, model_trained, test_loader, test=True)   #对测试集进行输出
        return model_trained, predict_label


