import numpy as np
import tensorflow as tf
from Bio import SeqIO
from tensorflow.keras import regularizers
def one_hot_encoding(sequence, amino_acids):
    encoding_dict = {aa: i for i, aa in enumerate(amino_acids)}
    encoded_sequence = [encoding_dict[aa] for aa in sequence]
    one_hot_sequence = tf.one_hot(encoded_sequence, len(amino_acids))
    return one_hot_sequence


# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, MaxPooling2D, Conv2D, LSTM, GRU, Bidirectional
def protein_classifier_cnn(input_size, num_classes, amino_acids):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(input_size, len(amino_acids))))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.02)))
    model.add(tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.02)))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid',kernel_regularizer=regularizers.l2(0.02)))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    # model.add(Conv1D(filters=27, kernel_size=(4), activation='relu',  input_shape=(input_size, len(amino_acids)))
    # model.add(MaxPooling1D(pool_size=(3)))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(2, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def protein_classifier_cnn_attention(input_size, num_classes, amino_acids):
    input_layer = tf.keras.Input(shape=(input_size, len(amino_acids)))
    conv_layer = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
    pooling_layer = tf.keras.layers.MaxPooling1D(pool_size=2)(conv_layer)
    attention_layer = tf.keras.layers.Attention()([pooling_layer, pooling_layer])
    flatten_layer = tf.keras.layers.Flatten()(attention_layer)
    dense_layer = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(
        flatten_layer)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(dense_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def protein_classifier_lstm(input_size, num_classes, amino_acids):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(64, input_shape=(input_size, len(amino_acids))))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

def protein_classifier_cnn_protvec(input_size, num_classes, amino_acids, embedding_dim):
    input_tensor = tf.keras.Input(shape=(input_size,), dtype=tf.int32)
    embedding_layer = tf.keras.layers.Embedding(input_dim=len(amino_acids), output_dim=embedding_dim)(input_tensor)
    embedding_output = tf.keras.layers.Flatten()(embedding_layer)
    # Convolutional Layers
    conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(embedding_layer)
    conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(conv1)
    # Attention Layer
    attention_layer = tf.keras.layers.Attention()([conv2, conv2])
    # Flatten and Dense Layers
    flatten = tf.keras.layers.Flatten()(attention_layer)
    dense = tf.keras.layers.Dense(128, activation='relu')(flatten)
    # Output Layer
    output_tensor = tf.keras.layers.Dense(num_classes, activation='softmax')(dense)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model

def protein_classifier_cnn_lstm(input_size, num_classes, amino_acids):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu',
                                     input_shape=(input_size, len(amino_acids))))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=(3)))
    model.add(tf.keras.layers.Conv1D(filters=14, kernel_size=(2), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=(2)))
    model.add(tf.keras.layers.LSTM(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    return model

from tensorflow.keras.models import Model
def convolutional_transformer_model(input_shape, num_classes, num_conv_blocks, d_model, num_heads, ff_dim, dropout_rate):
    inputs = tf.keras.Input(shape=input_shape)

    # Convolutional Blocks
    x = inputs
    for _ in range(num_conv_blocks):
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Transformer Layer
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Adjust the shape of inputs tensor to match the shape of x tensor
    cropped_inputs = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    cropped_inputs = tf.keras.layers.Dense(d_model)(cropped_inputs)
    cropped_inputs = tf.keras.layers.Reshape((1, d_model))(cropped_inputs)
    cropped_inputs = tf.tile(cropped_inputs, [1, x.shape[1], 1])

    x = tf.keras.layers.Add()([x, cropped_inputs])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(ff_dim, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(d_model)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Classification Layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# fasta_file = "../data/Arabidopsis/A.thaliana-all.fa"
# labels_file = "../data/Arabidopsis/A.thaliana-all-labels.txt"
fasta_file ="../data/Training/training-all.fa"
labels_file ="../data/Training/training-all-labels.txt"
protein_sequences = []
seqcount = 0
for record in SeqIO.parse(fasta_file, "fasta"):
    protein_sequences.append(str(record.seq))
    seqcount=seqcount +1
# print(seqcount)

labels = []
labelcount=0
with open(labels_file, "r") as labels_file:
    for line in labels_file:
        label = line.strip()
        labels.append(label)
        labelcount= labelcount + 1
# print(labelcount)

# 检查样本数量是否一致
if len(protein_sequences) != len(labels):
    raise ValueError("The number of protein sequences and labels does not match.")
# seqlen=0
# for item in protein_sequences:
#     seqlen=len(item)
#     print(seqlen)

max_sequence_length = max(len(str(sequence)) for sequence in protein_sequences)
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

input_size = max_sequence_length
num_classes = len(set(labels))  # 标签类别数量
input_shape = (101, 32)
num_conv_blocks = 2
d_model = 64
num_heads = 4
ff_dim = 128
dropout_rate = 0.2
# 对齐序列
import csv
padded_sequences = []
orginal_sequences = []
for sequence in protein_sequences:
    sequence=sequence.replace('X','')
    orginal_sequences.append(sequence)
    one_hot_sequence = one_hot_encoding(sequence, amino_acids)
    sequence_length = len(sequence)
    padding_length = max_sequence_length - sequence_length
    padded_sequence = tf.pad(one_hot_sequence, [[0, padding_length], [0, 0]])
    padded_sequences.append(padded_sequence)
# with open("../data/Training/padded_sequences.csv",'w',newline='') as file:
#     writer = csv.writer(file)
#     for row in orginal_sequences:
#         writer.writerow([row])

# inputs = tf.stack(padded_sequences)

# inputs = np.concatenate(padded_sequences, axis=0)
# inputs = tf.expand_dims(inputs, axis=-1)  #调整输入数据的形状，添加通道维度
from keras.callbacks import EarlyStopping
# 创建提前停止的回调
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
# 定义模型
# model = convolutional_transformer_model(input_size, num_classes, amino_acids)
# 编译模型
# 构建CNN-Transformer模型
# input_shape = (sequence_length, 32)
# model = convolutional_transformer_model(input_shape, num_classes, num_conv_blocks, d_model, num_heads, ff_dim, dropout_rate)

# model = convolutional_transformer_model(input_shape, num_classes, num_conv_blocks, d_model, num_heads, ff_dim, dropout_rate)
model = protein_classifier_cnn(input_size, num_classes, amino_acids)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 划分训练集和验证集
from sklearn.model_selection import train_test_split

# 将 TensorFlow 张量转换为 NumPy 数组
inputs_np = np.array(padded_sequences)
labels_np = np.array(labels)
train_inputs, val_inputs, train_labels, val_labels = train_test_split(inputs_np, labels_np, test_size=0.3, random_state=42)
# 将标签转换为整数编码
label_encoding = {label: i for i, label in enumerate(set(labels))}
train_labels = np.array([label_encoding[label] for label in train_labels])
val_labels = np.array([label_encoding[label] for label in val_labels])

# 训练模型
history = model.fit(train_inputs, train_labels, validation_data=(val_inputs, val_labels), epochs=10, batch_size=32)
#保存模型
model.save("protein_classifier_model.h5")

import matplotlib.pyplot as plt

# 提取训练过程中的损失值和准确度
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# 绘制损失曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确度曲线
plt.subplot(1, 1, 1)
plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Train')
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()

# 计算AUC值
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
output = model.predict(val_inputs)
auc = roc_auc_score(val_labels, output[:, 1])  # 假设正类的概率在output的第二列

# 计算ROC曲线的假正类率（FPR）和真正类率（TPR）
fpr, tpr, thresholds = roc_curve(val_labels, output[:, 1])

# 绘制AUC曲线
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label='CNN (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()

plt.tight_layout()
plt.show()

# 模型评估
test_loss, test_accuracy = model.evaluate(val_inputs, val_labels)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

#预测
output = model.predict(val_inputs)
predicted_labels = [list(label_encoding.keys())[np.argmax(pred)] for pred in output]
true_labels = [list(label_encoding.keys())[label] for label in val_labels]



# 打印分类报告
from sklearn.metrics import classification_report
report = classification_report(true_labels, predicted_labels, zero_division=1)
print("Classification Report:")
print(report)
