import tensorflow as tf
from tensorflow.keras.layers import Dropout,LSTM,Dense
from tensorflow.keras import Sequential

def lstm_model(fea_dim):
    model = Sequential()
    # model.add(Dropout(0.25))
    model.add(LSTM(64, input_shape=(fea_dim,4), recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(64,dropout=0.25,recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model