import os
import numpy as np
import pandas as pd
import tensorflow as tf
# feather 'Can only convert 1-dimensional array values'
# import feather
import pickle
import logging
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras import applications
from PIL import Image
from pathlib import Path
from pandas import DataFrame
# from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
matplotlib.use( 'TkAgg' )
logging._warn_preinit_stderr = 0
print(tf.executing_eagerly())
tf.debugging.set_log_device_placement(True)
print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())


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
            value = 'cleaned'
            imgname = str(filename)
            cid_value_imgname.append([cid, value, imgname])
        for filename in dirty_dishes_folder.glob('**/*.jpg'):
            cid = os.path.splitext(Path(filename).name)[0]
            value = 'dirty'
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
    def test_loop(model, test_folder, width, height, out_filename):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        # print(test_folder)
        test_generator = test_datagen.flow_from_directory(
            directory=test_folder,
            target_size=(height, width),
            class_mode=None,
            color_mode="rgb",
            shuffle = False,
            batch_size=6)
        # print(test_generator.n)
        STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
        predict = model.predict_generator(test_generator,
                                          steps=STEP_SIZE_TEST)
        print(predict)
        d = {'id': [n for n in range(len(predict))], 
             'label': ['dirty' if x > 0.5 else 'cleaned' for x in predict]}
             # 'label': ['dirty' if np.argmax(x) == 1 else 'clean' for x in predict]}
        result_df = DataFrame(data=d)
        result_df.to_csv(out_filename, index=False)
        print(result_df)
        # return result_df

    @staticmethod
    def train_model(train_generator, valid_generator, width, height, epochs_to_train):
        print('Use child methods!')
    
    def get_model(self, datagen):
        self.split_data(size=0.2, shuffle_bool=True)
        train_generator = datagen.flow_from_dataframe(
            dataframe=self.train_df,
            directory=None,
            color_mode='rgb',
            x_col="imgname",
            y_col="label",
            shuffle=True,
            class_mode="binary",
            # class_mode="sparse",
            # class_mode ='categorical',
            # classes = ['0', '1'],
            target_size=(self.height, self.width),
            batch_size=32)
        valid_generator = datagen.flow_from_dataframe(
            dataframe=self.valid_df,
            directory=None,
            color_mode='rgb',
            x_col="imgname",
            y_col="label",
            shuffle=True,
            class_mode="binary",
            # class_mode="sparse",
            # class_mode='categorical',
            # classes = ['0', '1'],
            target_size=(self.height, self.width),
            batch_size=8)
        '''
        for item in train_generator:
            print(item)
        print(train_generator.class_indices)
        print(train_generator.class_indices)
        '''
        _, model = self.train_model(train_generator, valid_generator, self.width, self.height, self.epocs)
        return model

class DishAnn(Ann):
    def __init__(self, df, width, height, epochs):
        super().__init__(df)
        self.epocs = epochs
        self.width = width
        self.height = height

    @staticmethod
    def train_model(train_generator, valid_generator, width, height, epochs_to_train):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(width, height, 3)))
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
        model.add(Dense(1, activation='sigmoid'))
        # model.add(Dense(2, activation='softmax'))
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, decay=0.0001), metrics=['accuracy'])
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, decay=0.0001), metrics=['accuracy'])
        STEP_SIZE_TRAIN = 100
        STEP_SIZE_VALID = 80
        # STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        # STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      validation_data=valid_generator,
                                      validation_steps=STEP_SIZE_VALID,
                                      epochs=epochs_to_train)
        # print(history.history)
        return history, model

class DishAnnInceptionResNetV2(Ann):
    def __init__(self, df, width, height, epochs):
        super().__init__(df)
        self.epocs = epochs
        self.width = width
        self.height = height

    @staticmethod
    def train_model(train_generator, valid_generator, width, height, epochs_to_train):
        '''
        model = applications.InceptionResNetV2(weights='imagenet', 
                                               include_top=False, 
                                               input_shape=(width, height, 3))
        '''
        model = applications.DenseNet121(weights='imagenet', 
                                      include_top=False, 
                                      input_shape=(width, height, 3))
        model.trainable = False
        model_out = model.output
        out_array = Flatten()(model_out)
        dense_out = Dense(1024, activation="relu")(out_array)
        # drop_out = Dropout(0.5)(dense_out)
        # predictions = Dense(2, activation='softmax')(drop_out)
        # predictions = Dense(2, activation='softmax')(dense_out)
        predictions = Dense(1, activation='sigmoid')(dense_out)
        model = Model(inputs=model.input, outputs=predictions)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, decay=0.0001), metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, decay=0.0001), metrics=['accuracy'])
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
        STEP_SIZE_TRAIN = 100
        STEP_SIZE_VALID = 80
        # STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        # STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      validation_data=valid_generator,
                                      validation_steps=STEP_SIZE_VALID,
                                      epochs=epochs_to_train)
        # print(history.history)
        return history, model

'''
def test():
    train_folder = Path(os.getcwd(), 'plates', 'train')
    test_folder = Path(os.getcwd(), 'plates', 'test')
    test_train_folder = Path(os.getcwd(), 'plates', 'train')
    train_df_name = Path(os.getcwd(), 'train_dishes_df.file')
    db_inst = Db()
    db_inst.create_df(train_folder)
    db_inst.save_dataframe(train_df_name)
    db_inst.read_dataframe(train_df_name)
    width, height = db_inst.get_image_size(0)
    print(width, height)
    # nn_inst = DishAnn(db_inst.df, 240, 240, 10)
    image_size = 224
    epocs = 5
    csv_filename = Path(os.getcwd(), 'BDAIRNV_BG_1.csv')
    train_csv_filename = Path(os.getcwd(), 'train.csv')
    # datagen = ImageDataGenerator(rescale=1. / 255)
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )
    nn_inst = DishAnn(db_inst.df, image_size, image_size, epocs)
    # nn_inst = DishAnnInceptionResNetV2(db_inst.df, image_size, image_size, epocs)
    model = nn_inst.get_model(datagen)
    nn_inst.test_loop(model, test_folder, image_size, image_size, csv_filename)
    nn_inst.test_loop(model, test_train_folder, image_size, image_size, train_csv_filename)
    # print(db_isnst.df)
    # db_isnst.show_train_image(0)


if __name__ == "__main__":
    test()
'''