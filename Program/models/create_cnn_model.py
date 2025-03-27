import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout


def cnn_model(fea_dim):
    filter_sizes = [3, 4, 5]
    num_filters = 64
    num_classes = 2
    kernel_size=5
    model = tf.keras.Sequential()
    model.add(Dropout(0.5))

    model.add(tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=(fea_dim, )))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(filters=num_filters*2, kernel_size=kernel_size, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(filters=num_filters*4, kernel_size=kernel_size, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    # model.add(Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model