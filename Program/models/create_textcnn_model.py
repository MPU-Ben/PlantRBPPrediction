import tensorflow as tf
from keras.src.optimizers import Adam
from keras import regularizers
from tensorflow.keras.layers import (Dropout,Attention,Flatten,
                                     Conv1D,GlobalMaxPooling1D,Reshape,Dense)

def textcnn_model(fea_dim):
    filter_sizes = [3, 4, 5]
    num_filters = 64
    dropout_rate = 0.5
    num_classes = 2
    L2_reg_lambda = 0.001
    learning_rate = 0.001
    inputs = tf.keras.Input(shape=(fea_dim, 1), dtype='float32')

    pooled_outputs = []
    for filter_size in filter_sizes:
        conv = Conv1D(
            filters=num_filters,
            kernel_size=filter_size,
            activation='relu',
            kernel_regularizer=regularizers.l2(L2_reg_lambda)
        )(inputs)
        pool = GlobalMaxPooling1D()(conv)
        pooled_outputs.append(pool)

    if len(filter_sizes) > 1:
        outputs = tf.keras.layers.concatenate(pooled_outputs)
    else:
        outputs = pooled_outputs[0]

    # attention
    outputs = Reshape((len(filter_sizes), num_filters))(outputs)
    query = outputs  # query 与 value 相同
    value = outputs
    attention = Attention()([query, value])
    attention = Flatten()(attention)

    outputs = Dropout(dropout_rate)(attention)
    outputs = Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=regularizers.l2(L2_reg_lambda)
    )(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model