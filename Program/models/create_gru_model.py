import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense,GRU,Attention,Input
from tensorflow.python.keras import regularizers


def gru_model(fea_dim):
    # num_classes = 2
    #
    # model = tf.keras.Sequential()
    # model.add(Dropout(0.25))
    # model.add(GRU(64, input_shape=(fea_dim, 1)))
    # # 添加注意力机制层
    # model.add(Attention())
    #
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    # model.add(Dropout(0.25))
    # model.add(Dense(num_classes, activation='softmax'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # return model
    # num_classes = 2
    #
    # inputs = tf.keras.Input(shape=(20,fea_dim))
    # x = Dropout(0.25)(inputs)
    # x = Reshape((-1, 20))(x)  # 调整输入形状
    # x = GRU(64, return_sequences=False)(x)
    #
    # # Attention layer
    # attention = Attention()([x, x])
    #
    # x = Dense(128, activation='relu')(attention)
    # x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.02))(x)
    # x = Dropout(0.25)(x)
    # outputs = Dense(num_classes, activation='softmax')(x)
    #
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # return model
    num_classes = 2

    inputs = Input(shape=(fea_dim, 1))
    x = Dropout(0.25)(inputs)
    x = GRU(64, return_sequences=False)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.02))(x)
    x = Dropout(0.25)(x)

    # 添加注意力层
    attention = Attention()([x, x])  # 将中间层的输出作为query和value传递给注意力层

    output = Dense(num_classes, activation='softmax')(attention)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model