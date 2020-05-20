import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import word2vec, fasttext
import gc

class Config:
    train_log_path = 'data/train_preliminary/click_log.csv'
    test_log_path = 'data/test/click_log.csv'
    train_ad_path = 'data/train_preliminary/ad.csv'
    test_ad_path = 'data/test/ad.csv'
    train_log_temp_path = 'data/temp/train_log.csv'      #merge ad info
    test_log_temp_path = 'data/temp/test_log.csv'
    sequence_creative_path = 'data/temp/creative_seq.txt'
    sequence_ad_path = 'data/temp/ad_seq.txt'
    sequence_product_path = 'data/temp/product_seq.txt'
    embedding_creative_w2v_path = 'data/embedding/creative_w2v.txt'
    embedding_creative_fasttext_path = 'data/embedding/creative_fasttext.txt'
    embedding_ad_w2v_path = 'data/embedding/ad_w2v.txt'
    embedding_ad_w2v_fasttext_path = 'data/embedding/ad_fasttext.txt'
    embedding_product_w2v_path = 'data/embedding/product_w2v.txt'
    embedding_product_w2v_fasttext_path = 'data/embedding/product_fasttext.txt'



class W2V:
    def __init__(self, config):
        #self.creative_path = config.sequence_creative_path
        self.path_dict = {'creative': config.sequence_creative_path}
        self.w2v_embedding_path_dict = {'creative': config.embedding_creative_w2v_path}
        self.fasttext_embedding_path_dict = {'creative': config.embedding_creative_fasttext_path}
        self.embedding_size = 100
        self.sg = 1                      #{0:cbow, 1:skip-gram}
        self.iter = 10
        self.window = 5
        self.workers = 5
        self.hs = 1
        self.min_count = 5

    def read_data(self, data_type):
        print("read data...")
        path = self.path_dict[data_type]
        sentences = []
        with open(path, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                word_list = line.strip('\n').split()
                sentences.append(word_list[1:])                #第一个值表示用户id，删除
        print("read finished")
        return sentences

    def train_model_w2v(self, data_type):
        sentences = self.read_data(data_type)
        save_path = self.w2v_embedding_path_dict[data_type]
        print("training w2v model...")
        model = word2vec.Word2Vec(sentences,
                                  min_count=self.min_count, iter=self.iter, sg=self.sg, window=self.window,
                                  size=self.embedding_size, workers=self.workers, hs=self.hs)
        model.wv.save_word2vec_format(save_path, binary=False)
        print("saved the w2v")


    def train_model_fasttext(self, data_type):
        sentences = self.read_data(data_type)
        save_path = self.fasttext_embedding_path_dict[data_type]
        print("training fasttext model...")
        model = fasttext.FastText(sentences,
                                  min_count=self.min_count, iter=self.iter, sg=self.sg, window=self.window,
                                  size=self.embedding_size, workers=self.workers, hs=self.hs,
                                  min_n=3, max_n=6)
        model.wv.save_word2vec_format(save_path, binary=False)
        print("saved the fasttext vector")


def generate_sequence(config, data_type):
    train_log = pd.read_csv(config.train_log_temp_path)
    test_log = pd.read_csv(config.test_log_temp_path)
    log_data = pd.concat([train_log, test_log], axis=0, ignore_index=True)
    del train_log, test_log
    gc.collect()
    log_data[data_type] = log_data[data_type].apply(lambda x: str(x))
    print("sort data...")
    log_data = log_data.sort_values(by=['user_id', 'time'])
    #print(train_log.head(20))
    #存储
    if data_type == 'ad_id':
        path = config.sequence_ad_path
    elif data_type == 'product_id':
        path = config.sequence_product_path
    else:
        raise ValueError("some thing wrong")
    with open(path, 'w', encoding='utf-8') as fw:
        pre = log_data.iloc[0]['user_id']
        creative_list = [str(pre)]     #第一位表示user_id
        for _, row in tqdm(log_data.iterrows()):
            if pre != row['user_id']:
                fw.write(' '.join(creative_list) + '\n')
                creative_list = [str(row['user_id']), row[data_type]]
                pre = row['user_id']
            else:
                creative_list.append(row[data_type])
        fw.write(' '.join(creative_list))

if __name__ == "__main__":
    # #step1  生成sequence数据
    config = Config()
    generate_sequence(config, 'product_id')
    print("finished")
    generate_sequence(config, 'ad_id')
    # #step2  训练w2v
    # data_type = 'creative'
    # config = Config()
    # w2v = W2V(config)
    # #w2v.train_model_fasttext(data_type)
    # w2v.train_model_w2v(data_type)
