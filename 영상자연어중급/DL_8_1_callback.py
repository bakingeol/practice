# DL_8_1_callback.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection, preprocessing
import matplotlib.pyplot as plt


# callback : 호출을 위임


def make_xy_2():
    df = pd.read_excel('data/BostonHousing.xls')

    x = df.values[:, :-2]
    y = df.values[:, -2:-1]

    return x, y


def show_history(history):
    print(history)
    print(history.history)
    print(type(history.history))
    print(history.history.keys())
    # dict_keys(['loss', 'mae'])
    # dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])

    # 퀴즈
    # history 객체에 들어있는 값을 그래프로 그려보세요

    # 퀴즈
    # 새롭게 추가한 밸리데이션 데이터까지 포함해서 그래프를 그려보세요
    plt.subplot(1, 2, 1)
    plt.title('loss')
    plt.plot(history.history['loss'], 'r', label='train')
    plt.plot(history.history['val_loss'], 'g', label='valid')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('mae')
    plt.plot(history.history['mae'], 'r', label='train')
    plt.plot(history.history['val_mae'], 'g', label='valid')
    plt.legend()
    plt.show()


def model_callback():
    x, y = make_xy_2()
    x = preprocessing.scale(x)

    data = model_selection.train_test_split(x, y, train_size=0.7, shuffle=False)
    x_train, x_test, y_train, y_test = data

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[13]))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.SGD(0.01),
                  loss=tf.keras.losses.mse,
                  metrics='mae')

    history = tf.keras.callbacks.History()

    # 퀴즈
    # 페이션스가 5일 때 어얼리 스토핑을 여러분 결과에 대해 분석하세요
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=5,
                                                  verbose=1)
    model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=2,
              validation_data=(x_test, y_test),
              callbacks=[history, early_stop])

    print(model.evaluate(x_test, y_test, verbose=0))
    model.save('data/model_boston.h5')
    # Epoch 87: early stopping
    # [300.6898498535156, 13.384793281555176]

    # history = model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=2,
    #                     validation_data=(x_test, y_test))

    # show_history(history)


def model_callback_2():
    x, y = make_xy_2()
    x = preprocessing.scale(x)

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[13]))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.SGD(0.01),
                  loss=tf.keras.losses.mse,
                  metrics='mae')

    file_path = 'model/boston_{epoch:02d}_{val_mae:.2f}.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path,
                                                    monitor='val_mae',
                                                    save_best_only=True,
                                                    verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.9,
                                                     patience=2,
                                                     verbose=1)

    model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=2,
              validation_data=(x_test, y_test),
              callbacks=[checkpoint, reduce_lr])


# 퀴즈
# model_callback 함수에서 저장한 모델을 읽어와서
# evaluate 함수로 결과를 확인하세요
def load_model_callback():
    x, y = make_xy_2()
    x = preprocessing.scale(x)

    data = model_selection.train_test_split(x, y, train_size=0.7, shuffle=False)
    _, x_test, _, y_test = data

    model = tf.keras.models.load_model('data/model_boston.h5')
    print(model.evaluate(x_test, y_test, verbose=0))
    # [300.6898498535156, 13.384793281555176]


# model_callback()
# load_model_callback()

model_callback_2()
