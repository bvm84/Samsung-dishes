import os
import numpy as np
import pandas as pd
import tensorflow as tf
# feather 'Can only convert 1-dimensional array values'
# import feather
import pickle
import logging
import matplotlib
# matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
logging._warn_preinit_stderr = 0
print(tf.executing_eagerly())


class Db():
    def __init__(self):
        self._df = None

    @staticmethod
    def load_image(image_name):
        img = Image.open(image_name)
        img.load()
        data = np.asarray(img, dtype="int32")
        return data

    def create_df(self, train_dishes_folder):
        cid_value_imgname = []
        clean_dishes_folder = train_dishes_folder.joinpath('cleaned')
        dirty_dishes_folder = train_dishes_folder.joinpath('dirty')
        for filename in clean_dishes_folder.glob('**/*.jpg'):
            cid = os.path.splitext(Path(filename).name)[0]
            value = 0
            imgname = str(filename)
            cid_value_imgname.append([cid, value, imgname])
        for filename in dirty_dishes_folder.glob('**/*.jpg'):
            cid = os.path.splitext(Path(filename).name)[0]
            value = 1
            imgname = str(filename)
            cid_value_imgname.append([cid, value, imgname])
        self._df = DataFrame(cid_value_imgname, columns=['cid', 'label', 'imgname'])

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

    def get_image_size(self, n):
        im = Image.open(self.df.iloc[n]['imgname'])
        width, height = im.size
        return width, height

# class FeatureExtractor():


class Ann():
    def __init__(self, df):
        self.df = df
        self.train_df = None
        self.valid_df = None

    def split_data(self, size=0.2, shuffle_bool=True):
        self.train_df, self.valid_df = train_test_split(self.df, test_size=size, shuffle=shuffle_bool)

    @staticmethod
    def save_model(model, model_name):
        model.save(
            model_name,
            overwrite=True,
            include_optimizer=True
        )

    @staticmethod
    def load_model(model_name):
        model = tf.keras.models.load_model(model_name)
        return model

    @staticmethod
    def predict_model(model, array):
        predicted = model.predict(np.expand_dims(array, axis=0))
        return np.round(np.squeeze(predicted))

    @staticmethod
    def test_model_loop(model, test_folder, width, height):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
            directory=test_folder,
            target_size=(height, width),
            color_mode="rgb",
            batch_size=8)
        STEP_SIZE_TEST = test_generator.n // test_generator.n.batch_size
        predict = model.predict_generator(test_generator,
                                          steps=STEP_SIZE_TEST)
        print(predict)
        '''
        result_df = DataFrame(data=d)
        result_df.to_excel("output.xlsx")
        print(result_df)
        return result_df
        '''


class DishAnn(Ann):
    def __init__(self, df, width, height, epochs):
        super.__init__(df)
        self.epocs = epochs
        self.width = width
        self.height = height

    @staticmethod
    def train_dish_model(train_generator, valid_generator, epochs_to_train):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      validation_data=valid_generator,
                                      validation_steps=STEP_SIZE_VALID,
                                      epochs=epochs_to_train)
        # print(history.history)
        return history, model

    def get_class_model(self):
        self.split_data(self, size=0.2, shuffle_bool=True)
        datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = datagen.flow_from_dataframe(
            dataframe=self.train_df,
            directory=None,
            color_mode='rgb',
            x_col="imgname",
            y_col="label",
            class_mode="binary",
            target_size=(self.height, self.width),
            batch_size=8)
        valid_generator = datagen.flow_from_dataframe(
            dataframe=self.valid_df,
            directory=None,
            color_mode='rgb',
            x_col="imgname",
            y_col="label",
            class_mode="binary",
            target_size=(self.height, self.width),
            batch_size=8)
        _, model = self.train_cat_model(train_generator, valid_generator, self.epocs)
        return model


def test():
    train_folder = Path(os.getcwd(), 'plates', 'train')
    test_folder = Path(os.getcwd(), 'test')
    train_df_name = Path(os.getcwd(), 'train_dishes_df.file')
    db_inst = Db()
    # db_inst.create_df(train_folder)
    # db_inst.save_dataframe(train_df_name)
    db_inst.read_dataframe(train_df_name)
    width, heigth = db_inst.get_image_size(0)
    print(width, heigth)
    # db_inst.show_train_image(0)
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

