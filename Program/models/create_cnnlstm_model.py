import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras import regularizers


def cnnlstm_model(fea_dim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu',
                                     input_shape=(fea_dim,1)))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=(3)))
    model.add(tf.keras.layers.Conv1D(filters=14, kernel_size=(2), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=(2)))
    model.add(tf.keras.layers.LSTM(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model