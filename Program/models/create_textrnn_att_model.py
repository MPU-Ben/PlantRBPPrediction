import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout, SimpleRNN, Dense,Attention,Flatten
from tensorflow.keras.models import Model


def text_rnn_attention_model(fea_dim):
    num_units = 32
    num_classes = 2

    inputs = tf.keras.Input(shape=(fea_dim,4))
    rnn_layer = SimpleRNN(units=num_units, activation='relu', return_sequences=True)(inputs)
    attention = Attention()([rnn_layer, rnn_layer])
    attention = Flatten()(attention)

    dense_layer = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(attention)
    dense_layer = Dense(128, activation='relu')(dense_layer)
    dense_layer = Dense(64, activation='relu')(dense_layer)
    outputs = Dense(num_classes, activation='sigmoid')(dense_layer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model