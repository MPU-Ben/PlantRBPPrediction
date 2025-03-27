import tensorflow as tf
from tensorflow.keras.layers import (Dropout,Attention,Flatten,
                                     Conv1D,GlobalMaxPooling1D,Reshape,Dense)

def textcnn_model(fea_dim):

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    embedding_dim = 128
    filter_sizes = [3, 4, 5]
    num_filters = 64
    dropout_rate = 0.5
    num_classes = 2

    inputs = tf.keras.Input(shape=(fea_dim, 1), dtype='float32')

    pooled_outputs = []
    for filter_size in filter_sizes:
        conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(inputs)
        pool = GlobalMaxPooling1D()(conv)
        pooled_outputs.append(pool)

    if len(filter_sizes) > 1:
        outputs = tf.keras.layers.concatenate(pooled_outputs)
    else:
        outputs = pooled_outputs[0]

    # 添加注意力机制层
    outputs = Reshape((len(filter_sizes), num_filters))(outputs)
    query = outputs  # query 与 value 相同
    value = outputs
    attention = Attention()([query, value])
    attention = Flatten()(attention)

    outputs = Dropout(dropout_rate)(attention)
    # outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(num_classes, activation='softmax')(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model