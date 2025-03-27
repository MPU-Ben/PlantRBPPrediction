import pandas as pd
import numpy as np
import tensorflow as tf
from keras.src.utils import np_utils
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, matthews_corrcoef, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.utils.np_utils import to_categorical
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import models.create_textcnn_model
import models.create_gru_model
from tensorflow.keras.callbacks import TensorBoard
# Assuming you have the dataset in a CSV file
data = pd.read_csv('../files/optimumAllTrainingDataset_F1.csv')
# Split the dataset into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train,X_val,y_train,y_val=train_test_split(X,y,shuffle=True,test_size=0.5,stratify=y)
# Convert labels to one-hot encoding
testing_data = pd.read_csv('../files/optimumIndependenceTestingDataset.csv')
X_independent_test = data.iloc[:, :-1].values
y_independent_test = data.iloc[:, -1].values
# TextCNN model
embedding_dim = 100
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
vocab_size = len(amino_acids)
sample_num = X.shape[0]
num_classes = len(set(y))  # 标签类别数量
feature_dim = X.shape[1]
num_epochs=50
num_batch_size = 64

textcnn_model = models.create_textcnn_model.textcnn_model(feature_dim)

textcnn_aucs = []
textcnn_accs = []

reshaped_X_train = np.zeros((X_train.shape[0], X_train.shape[1], 1))
reshaped_X_train[:, :, 0] = X_train[:, :]
y_train_encoded = to_categorical(y_train, num_classes)
y_val_encoded = to_categorical(y_val, num_classes)
y_independent_test_encoded = to_categorical(y_independent_test, num_classes)
print("start training with TextCNN")
# y_train_encoded = to_categorical(num_classes)
history = textcnn_model.fit(reshaped_X_train, y_train_encoded, epochs=num_epochs, batch_size=num_batch_size,
                            verbose=0, validation_split=0.2)#,validation_data=(X_val, y_val_encoded),callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
train_key = history.history.keys()
print(train_key)
train_loss = history.history['loss']
print("train_loss",train_loss)
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
print("train_accuracy:",train_accuracy)
val_accuracy= history.history['val_accuracy']
print("END training with TextCNN")

# print("start testing with TextCNN")
# textcnn_probabilities = textcnn_model.predict(X_independent_test)[:,1]
# print("end testing with TextCNN")
print("start evaluate TextCNN")
# 预测独立测试集的标签
predictions = textcnn_model.predict(X_independent_test)
predicted_classes = np.argmax(predictions, axis=1)
# 真实标签
true_classes = np.argmax(y_independent_test_encoded, axis=1)
# 计算评价指标
acc = accuracy_score(true_classes, predicted_classes)
mcc = matthews_corrcoef(true_classes, predicted_classes)
cm = confusion_matrix(true_classes, predicted_classes)
# 灵敏度 (Sensitivity) 和特异性 (Specificity) 计算
tn, fp, fn, tp = cm.ravel()
sn = tp / (tp + fn)  # 灵敏度
sp = tn / (tn + fp)  # 特异性
f1 = f1_score(true_classes, predicted_classes, average='weighted')
# 打印结果
print("Accuracy (ACC):", acc)
print("Matthews Correlation Coefficient (MCC):", mcc)
print("Sensitivity (SN):", sn)
print("Specificity (SP):", sp)
print("F1 Score:", f1)
# text_result = textcnn_model.evaluate(X_independent_test,y_independent_test_encoded)
# print(text_result)
print("end evaluate TextCNN")
#绘图
import matplotlib.pyplot as plt

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
# test_loss, test_accuracy = textcnn_model.evaluate(X_test,y_test_encoded)
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test Accuracy: {test_accuracy:.4f}")
# #预测
# output = textcnn_model.predict(X_test)
# predicted_labels = [list(label_encoding.keys())[np.argmax(pred)] for pred in output]
# true_labels = [list(label_encoding.keys())[np.argmax(label)] for label in y_test_encoded]
# # 打印分类报告
# from sklearn.metrics import classification_report
# report = classification_report(true_labels, predicted_labels, zero_division=1)
# print("Classification Report:")
# print(report)

# Perform 5-fold cross-validation
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# for train_index, test_index in kfold.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     reshaped_X_train = np.zeros((X_train.shape[0], X_train.shape[1], 1))
#     reshaped_X_train[:, :, 0] = X_train[:, :]
#     y_train_encoded = to_categorical(y_train,num_classes)
#     y_test_encoded = to_categorical(y_test, num_classes)
#
#     # TextCNN model training and prediction
#     print("start training with TextCNN")
#     # y_train_encoded = to_categorical(num_classes)
#     history = textcnn_model.fit(reshaped_X_train, y_train_encoded, epochs=num_epochs, batch_size = num_batch_size, verbose=0,)#callbacks=[print_batch_accuracy_callback]
#     train_key = history.history.keys()
#     print(train_key)
#     train_loss = history.history['loss']
#     print("train_loss",train_loss)
#     # val_loss = history.history['val_loss']
#     train_accuracy = history.history['accuracy']
#     print("train_accuracy:",train_accuracy)
#     textcnn_probabilities = textcnn_model.predict(X_test)[:,1]
#     print("END training with TextCNN")
#
#
#     y_test_binary = y_test #y_test_binary[:, 0, 1]
#
#     textcnn_fpr, textcnn_tpr, _ = roc_curve(y_test, textcnn_probabilities)
#
#     textcnn_auc = auc(textcnn_fpr, textcnn_tpr)
#
#     textcnn_aucs.append(textcnn_auc)
#
#
#     threshold = 0.8  # 阈值
#     # 将概率值转换为类别标签
#     textcnn_binary_predictions = [1 if prob >= threshold else 0 for prob in textcnn_probabilities]
#     textcnn_acc = accuracy_score(y_pred=textcnn_binary_predictions, y_true=y_test)
#     textcnn_accs.append(textcnn_acc)
# # Compute the mean AUC scores
# textcnn_mean_auc = np.mean(textcnn_aucs)
#
#
# fig = plt.figure()
# plt.plot(textcnn_fpr, textcnn_tpr, label=f'TextCNN (AUC = {textcnn_mean_auc:.2f})')
# plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k', label='Random')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
#
# # Set labels and title
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.title('ROC')
# plt.tick_params(axis='both', which='major', labelsize=14)  # 设置刻度标签大小
# plt.legend(loc="lower right", fontsize=12)  # 设置图例字体大小
# plt.grid(alpha=0.3)  # 添加网格线
# # plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
# plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.yaxis.grid(True)
#
# plt.boxplot([textcnn_accs],
#             patch_artist=True, vert=True, whis=True, showbox=True)
# ax.set_xticklabels(['textcnn'], fontsize=12)
# plt.xlabel('Classifiers', fontsize=12, fontweight='bold')
# plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
# plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
# plt.show()