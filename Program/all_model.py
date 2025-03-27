import numpy as np
import tensorflow as tf
from Bio import SeqIO
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout # , Dense, Flatten, Conv1D, MaxPooling1D, MaxPooling2D, Conv2D, LSTM, GRU, Bidirectional
def one_hot_encoding(sequence, amino_acids):
    encoding_dict = {aa: i for i, aa in enumerate(amino_acids)}
    encoded_sequence = [encoding_dict[aa] for aa in sequence]
    one_hot_sequence = tf.one_hot(encoded_sequence, len(amino_acids))
    return one_hot_sequence

def protein_classifier_bilstm(input_size, num_classes, amino_acids):
    model = tf.keras.Sequential()
    model.add(Dropout(0.25))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, input_shape=(input_size, len(amino_acids)))))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model
def protein_classifier_rnn(input_size, num_classes, amino_acids):
    model = tf.keras.Sequential()
    model.add(Dropout(0.25))
    model.add(tf.keras.layers.SimpleRNN(64, input_shape=(input_size, len(amino_acids))))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.25))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

def protein_classifier_gru(input_size, num_classes, amino_acids):
    model = tf.keras.Sequential()
    model.add(Dropout(0.25))
    model.add(tf.keras.layers.GRU(64, input_shape=(input_size, len(amino_acids))))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.25))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

def protein_classifier_cnn(input_size, num_classes, amino_acids):
    model = tf.keras.Sequential()
    model.add(Dropout(0.25))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(input_size, len(amino_acids))))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid',kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.25))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
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
    model.add(Dropout(0.25))
    model.add(tf.keras.layers.LSTM(64, input_shape=(input_size, len(amino_acids))))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
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

from keras.callbacks import EarlyStopping
# 创建提前停止的回调
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
# 定义模型
model = protein_classifier_cnn(input_size, num_classes, amino_acids)
# 编译模型
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 划分训练集和验证集
from sklearn.model_selection import train_test_split

# 将 TensorFlow 张量转换为 NumPy 数组
inputs_np = np.array(padded_sequences)
labels_np = np.array(labels)
train_inputs, val_inputs, train_labels, val_labels = train_test_split(inputs_np, labels_np, test_size=0.3, random_state=42)
# 将标签转换为整数编码
# label_encoding = {label: i for i, label in enumerate(set(labels))}
# train_labels = np.array([label_encoding[label] for label in train_labels])
# val_labels = np.array([label_encoding[label] for label in val_labels])
# 将标签转换为整数编码
label_encoding = {label: i for i, label in enumerate(set(labels))}
train_labels = tf.one_hot([label_encoding[label] for label in train_labels], num_classes)
val_labels = tf.one_hot([label_encoding[label] for label in val_labels], num_classes)
# 训练模型
history = model.fit(train_inputs, train_labels, validation_data=(val_inputs, val_labels), epochs=10, batch_size=64)

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
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Train')
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()

# 模型评估
test_loss, test_accuracy = model.evaluate(val_inputs, val_labels)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

#预测
output = model.predict(val_inputs)
predicted_labels = [list(label_encoding.keys())[np.argmax(pred)] for pred in output]
# true_labels = [list(label_encoding.keys())[label] for label in val_labels]
true_labels = [list(label_encoding.keys())[np.argmax(label)] for label in val_labels]
# 打印分类报告
from sklearn.metrics import classification_report
report = classification_report(true_labels, predicted_labels, zero_division=1)
print("Classification Report:")
print(report)
