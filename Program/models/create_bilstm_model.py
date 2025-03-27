import tensorflow as tf
from tensorflow.keras.layers import Dropout,Bidirectional,Dense,LSTM
from tensorflow.keras import Sequential

def bilstm_model(fea_dim):
    model = Sequential()
    # model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(64, input_shape=(fea_dim,4),recurrent_dropout=0.2,return_sequences=True)))
    model.add(Bidirectional(LSTM(64,recurrent_dropout=0.2,dropout = 0.25)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model