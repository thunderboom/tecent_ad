import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import copy

class Data_example():  #定义输入例子
    def __init__(self, user_id=None, creative_id=None, product_id=None, age=1, gender=1):
        self.user_id = user_id
        self.creative_id = creative_id
        self.product_id = product_id
        self.age = age
        self.gender = gender

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Reader():
    def __init__(self):
        self.creative_seq_path = 'data/temp/creative_seq.txt'
        self.product_seq_path = 'data/temp/product_seq.txt'
        self.train_path = 'data/train_preliminary/user.csv'

    def read_txt(self, path):
        text = []
        user_id = []
        with open(path, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                line = line.strip('\n').split()
                user_id.append(line[0])
                text.append(line[1:])
        return text, user_id

    def split(self, data):
        return data[:900000], data[900000:]

    def read_data(self):
         user_info = pd.read_csv(self.train_path)
         creative_text, user_id = self.read_txt(self.creative_seq_path)
         train_creative_data, test_creative_data = self.split(creative_text)
         train_user_id, test_user_id = self.split(user_id)
         age_list = user_info['age'].tolist()
         gender_list = user_info['gender'].tolist()
         train_data, test_data = [], []
         for user_id, creative, age, gender in zip(train_user_id, train_creative_data, age_list, gender_list):
             example = Data_example(user_id=user_id, creative_id=creative, age=age, gender=gender)
             train_data.append(example)
         for user_id, creative in zip(test_user_id, test_creative_data):
             example = Data_example(user_id=user_id, creative_id=creative)
             test_data.append(example)
         return train_data, test_data


class FeatureGeneration():
    def __init__(self): #定义embdding路径
        self.embedding_path_dict = {'creative_id':'data/embedding/creative_w2v.txt'}

    def load_word_dict(self, data_type):
        word_list = []
        path = self.embedding_path_dict[data_type]
        with open(path, 'r', encoding='utf-8') as fr:  # 定义矩阵
            for idx, line in enumerate(fr.readlines()):
                if idx != 0:
                    line = line.split()
                    word_list.append(line[0])
        word2id = {word: idx + 1 for idx, word in enumerate(word_list)}
        return word2id

    def load_embedding(self, data_type):
        path = self.embedding_path_dict[data_type]
        with open(path, 'r', encoding='utf-8') as fr:      #定义矩阵
            for idx, line in tqdm(enumerate(fr.readlines())):
                if idx == 0:
                    word_num, size = map(int, line.split())
                    embedding_matrix = np.zeros((word_num+1, size))
                else:
                    line = line.split()
                    embedding_matrix[idx, :] = np.array(line[1:], dtype=np.float)
        return embedding_matrix

    def convert_example_feature(self, examples, max_length):
        def sentence_feature(sentence, word2id, max_length):
            feature_sentence = []
            for idx, word in enumerate(sentence):
                try:
                    feature_sentence.append(word2id[word])
                except:
                    feature_sentence.append(0)
            true_length = len(feature_sentence)
            if true_length >= max_length:
                return feature_sentence[:max_length]
            else:
                feature_sentence.extend([0] * (max_length - true_length))
                return feature_sentence

        def gender_map(gender):
            if gender == 1:
                return 0
            else:
                return 1
        creative_word2id = self.load_word_dict('creative_id')
        feature_examples = []
        for example in tqdm(examples):
            example.creative_id = sentence_feature(example.creative_id, creative_word2id, max_length)
            example.gender = gender_map(example.gender)
            feature_examples.append(example)
        return feature_examples



class AdvData(Dataset):
    def __init__(self, data, config):
        self.data = data

    def __getitem__(self, index):
        temp_data = self.data[index]
        creative_id = np.array(temp_data.creative_id).astype(np.long)
        age = np.array(temp_data.age, dtype=int)
        gender = np.array(temp_data.gender, dtype=int)
        return creative_id, age, gender

    def __len__(self):
        return len(self.data)


def result_save(examples):
    user_ids = []
    for example in examples:
        user_ids.append(example.user_ids)
    return



#test
# class Config():
#     def __init__(self):
#         self.model_type = ''
#         self.data_type = ''
#         self.model_saved_path = 'model/' + self.model_type + '/' + self.data_type + '.bin'
#         self.hidden_size = 128
#         self.seed = 520
#         self.batch_size = 8
#         self.pattern = 'cross_validation'
#         self.model_saved = False
#         self.max_length = 15
#         self.require_grad = False  # 训练词向量
#         #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.val_size = 0.2
#
# if __name__ == "__main__":
#     reader = Reader()
#     config = Config()
#     print("reading data...")
#     train_data, test_data = reader.read_data()
#     print("transfer features...")
#     generation = FeatureGeneration()
#     train_data, test_data = generation.convert_example_feature(train_data, config.max_length), \
#                             generation.convert_example_feature(test_data, config.max_length)
#     train_dataset = AdvData(train_data, config)
#     train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
#     for i, (creative, age, gender) in enumerate(train_loader):
#         print(creative, age, gender)
#         break
    # print(test_data.user_id)
    # generation = FeatureGeneration()
    # word2id = generation.load_word_dict('creative_id')
    # sentence = train_data
    # print(generation.convert_example_feature(sentence, word2id))
    # print(generation.load_embedding('creative_id').shape)





