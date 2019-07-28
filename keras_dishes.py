import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
# feather 'Can only convert 1-dimensional array values'
# import feather
import pickle
import logging
import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from pandas import DataFrame, Series
from scipy.io import wavfile
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input, Conv1D, MaxPooling1D, Flatten, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
logging._warn_preinit_stderr = 0
print(tf.executing_eagerly())


class Db():
    def __init__(self):
        self._df = None

    @staticmethod
    def load_image(image_name):
        img = Image.open(image_name)
        img.load()
        data = np.asarray(img, dtype="int32" )
        return data  

    def create_df(self, train_dishes_folder):
        cid_value_imgname_data = []
        clean_dishes_folder = train_dishes_folder.joinpath('cleaned')
        dirty_dishes_folder = train_dishes_folder.joinpath('dirty')
        for filename in clean_dishes_folder.glob('**/*.jpg'):
            cid = os.path.splitext(Path(filename).name)[0]
            value = 0
            imgname = str(filename)
            data = self.load_image(filename)
            cid_value_imgname_data.append([cid, value, imgname, data])
        for filename in dirty_dishes_folder.glob('**/*.jpg'):
            cid = os.path.splitext(Path(filename).name)[0]
            value = 1
            imgname = str(filename)
            data = self.load_image(filename)
            cid_value_imgname_data.append([cid, value, imgname, data])
        self._df = DataFrame(cid_value_imgname_data, columns=['cid', 'value', 'imgname', 'data'])

    @property
    def df(self):
        return self._df

    @df.setter
    def set_df(self, df):
        if isinstance(df, pd.DataFrame):
            self._df = df
        else:
            print('Wrong dataframe')

    def save_dataframe(self, df_filename):
        try:
            with open(df_filename, 'wb') as f:
                pickle.dump(self._df, f)
        except Exception as e:
            logging.error(e)

    def read_dataframe(self, df_filename):
        try:
            with open(df_filename, 'rb') as f:
                self._df = pickle.load(f)
        except Exception as e:
            logging.error(e)

    @staticmethod
    def show_img(img):
        plt.imshow(img)
    
    def show_train_image(self, n):
        self.show_img(self.df.iloc[n]['data'])
        plt.show()


# class FeatureExtractor():

class WavAnn():
    def __init__(self, df):
        self.df = df
        self.train_df = None
        self.test_df = None
        self.ndf_train_x = None
        self.ndf_train_y = None
        self.ndf_test_x = None
        self.ndf_test_y = None
        self.xnscale_object = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
        self.ynscale_object = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)

    def split_data(self, size=0.2, shuffle_bool=True):
        self.train_df, self.test_df = train_test_split(self.df, test_size=size, shuffle=shuffle_bool)

'''
    def normalize_df(self):
        fetures_list = ['a' + str(x) for x in range(len(self.train_df['signal'].iloc[0]))]
        arrays_list = [arr for _, arr in self.train_df['signal'].iteritems()]
        var_df = DataFrame(data=arrays_list, columns=fetures_list)
        self.ndf_train_x = var_df.copy(deep=True)
        self.ndf_train_x[self.ndf_train_x.columns] = (
            self.xnscale_object.fit_transform(self.ndf_train_x[self.ndf_train_x.columns])
        )
        # print(var_df.iloc[0])
        # print(self.ndf_train_x.iloc[0])
        arrays_list = [arr for _, arr in self.test_df['signal'].iteritems()]
        var_df = DataFrame(data=arrays_list, columns=fetures_list)
        self.ndf_test_x = var_df.copy(deep=True)
        self.ndf_test_x[self.ndf_test_x.columns] = (
            self.xnscale_object.transform(self.ndf_test_x[self.ndf_test_x.columns])
        )
        # print(var_df.iloc[0])
        # print(self.ndf_test_x.iloc[0])
        self.ndf_train_y = pd.to_numeric(Series(self.ynscale_object.fit_transform(
            self.train_df['sys'].values.reshape(-1, 1)).flatten()))
        self.ndf_test_y = pd.to_numeric(Series(self.ynscale_object.transform(
            self.test_df['sys'].values.reshape(-1, 1)).flatten()))
        # print(self.ndf_train_y)
        # print(self.ndf_test_y)
'''
    @staticmethod
    def train_class_conv_model(x_train, y_train, epochs_to_train):
        x_train = np.expand_dims(x_train, axis=2)
        train_samle_length = len(x_train[0])
        print(x_train)
        print(train_samle_length)
        print(x_train.shape)
        model = Sequential()
        model.add(Conv1D(32, (3), input_shape=(16383, 1), activation='relu'))
        model.add(Conv1D(100, 10, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(160, 10, activation='relu'))
        model.add(Conv1D(160, 10, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='linear'))
        '''
        model_input = Input(shape=(16383, 1))
        model_conv_11 = Conv1D(filters=32, kernel_size=2, activation='relu')(model_input)
        model_pool_11 = MaxPooling1D(pool_size=2)(model_conv_11)
        model_conv_12 = Conv1D(filters=32, kernel_size=2, activation='relu')(model_pool_11)
        model_pool_12 = MaxPooling1D(pool_size=2)(model_conv_12)
        model_dropout = Dropout(0.5)(model_pool_12)
        # model_flatten_12 = Flatten()(model_pool_12)
        predict_out = Dense(1, activation='linear')(model_dropout)
        model = Model(inputs=model_input, outputs=predict_out)
        '''
        model.compile(loss='mean_squared_error', optimizer='sgd')
        history = model.fit(x_train, y_train, epochs=epochs_to_train)
        # print(history.history)
        return history, model

    def get_class_model(self):
        self.y_min = self.train_df['sys'].min()
        x = self.ndf_train_x.values.astype(dtype='float64')
        y = to_categorical(self.train_df['sys'].subtract(self.y_min).values)
        self.cat_n = len(y[0])
        _, model = self.train_cat_model(x, y, 20)
        return model

    def test_class_model(self, model):
        test_x = self.ndf_test_x.values.astype(dtype='float64')
        test_y = to_categorical(self.test_df['sys'].subtract(self.y_min).values, num_classes=self.cat_n)
        scores = model.evaluate(x=test_x, y=test_y, verbose=1)
        return scores

    def save_model(model, model_name):
        model.save(
            model_name,
            overwrite=True,
            include_optimizer=True
        )

    def load_model(model_name):
        model = tf.keras.models.load_model(model_name)
        return model

    @staticmethod
    def predict_cat_model(model, array):
        predicted = model.predict(np.expand_dims(array, axis=0))
        return np.round(np.squeeze(predicted))


    def test_cat_model_loop(self, model):
        predicted_list = []
        anoted_list = []
        for index, value in self.test_df['signal'].iteritems():
            anoted_list.append(self.test_df['sys'][index])
            print(value)
            narr = self.xnscale_object.transform(np.expand_dims(value, axis=0)).flatten()
            result = self.predict_cat_model(model, narr)
            predicted = np.argmax(result) + self.y_min
            predicted_list.append(predicted)
            print(predicted)
        d = {'cid': self.test_df['cid'].values, 'sysp': predicted_list,
             'sysa': anoted_list, 'diap': predicted_list, 'diaa': anoted_list}
        result_df = DataFrame(data=d)
        result_df.to_excel("output.xlsx")
        print(result_df)
        return result_df

def test():
    train_folder = Path(os.getcwd(), 'plates', 'train')
    test_folder = Path(os.getcwd(), 'test')
    train_df_name = Path(os.getcwd(), 'train_dishes_df.file')
    db_isnst = Db()
    # db_isnst.create_df(train_folder)
    # db_isnst.save_dataframe(train_df_name)
    db_isnst.read_dataframe(train_df_name)
    db_isnst.show_train_image(0)
    # df = db_isnst.df
    # print(df)
    # print(df.memory_usage(index=True, deep=True))
    '''
    ann = WavAnn(df)
    # ann.truncate_data(rows_to_store=10)
    ann.split_data()
    ann.normalize_df()
    # model = ann.get_reg_conv_model()
    model = ann.get_reg_model()
    d = ann.test_reg_model_loop(model)
    # d = ann.test_cat_model_loop(model)
    # d = ann.test_reg_conv_model_loop(model)
    xlsx_suf = '.xlsx'
    bname = PurePath(os.getcwd(), 'WavAnn_32s_reg_4994_2').with_suffix(xlsx_suf)
    data_sh_name = 'Data'
    graph_sh_name = 'Graphs'
    report_creator = Report(d)
    report_creator.create_tn_class_df()
    report_creator.create_histogram_df()
    report_creator.excel(bname, data_sh_name, graph_sh_name)
    '''


if __name__ == "__main__":
    test()

