import pandas as pd
import numpy as np
from keras.src.utils import np_utils
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_curve, auc, accuracy_score
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
data = pd.read_csv('../files/optimumAllDataset.csv')

# Split the dataset into features and labels
X = data.iloc[:, :-1].values
print(X.shape)
y = data.iloc[:, -1].values
# Convert labels to one-hot encoding

# TextCNN model
embedding_dim = 100
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
vocab_size = len(amino_acids)
sample_num = X.shape[0]
num_classes = len(set(y))  # 标签类别数量
feature_dim = X.shape[1]
num_epochs=10
num_batch_size = 64

textcnn_model = models.create_textcnn_model.textcnn_model(feature_dim)
gru_model = models.create_gru_model.gru_model(feature_dim)
# XGBoost model
xgb_model = XGBClassifier()

# Random Forest model
rf_model = RandomForestClassifier()
lda_model = LinearDiscriminantAnalysis(n_components = None, shrinkage = None, solver = 'lsqr')
# Initialize lists to store AUC values
textcnn_aucs = []
xgb_aucs = []
rf_aucs = []
lda_aucs = []
gru_aucs = []

lda_accs = []
gru_accs = []
textcnn_accs = []
# Perform 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    reshaped_X_train = np.zeros((X_train.shape[0], X_train.shape[1], 1))
    reshaped_X_train[:, :, 0] = X_train[:, :]
    y_train_encoded = to_categorical(y_train,num_classes)
    y_test_encoded = to_categorical(y_test, num_classes)

    # TextCNN model training and prediction
    print("start training with TextCNN")
    # y_train_encoded = to_categorical(num_classes)
    history = textcnn_model.fit(reshaped_X_train, y_train_encoded, epochs=num_epochs, batch_size = num_batch_size, verbose=0,)#callbacks=[print_batch_accuracy_callback]
    train_key = history.history.keys()
    print(train_key)
    train_loss = history.history['loss']
    print("train_loss",train_loss)
    # val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    print("train_accuracy:",train_accuracy)
    textcnn_probabilities = textcnn_model.predict(X_test)[:,1]
    print("END training with TextCNN")
    # # XGBoost model training and prediction
    # print("start training with XGB")
    # xgb_model.fit(X_train, y_train)
    # xgb_probabilities = xgb_model.predict_proba(X_test)[:, 1]
    # print("END training with XGB")
    # # Random Forest model training and prediction
    # print("start training with RF")
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    # rf_model.fit(X_train_scaled, y_train)
    # rf_probabilities = rf_model.predict_proba(X_test_scaled)[:, 1]
    # print("END training with RF")
    # # LDA model training and prediction
    # print("start training with LDA")
    # lda_model.fit(X_train, y_train)
    # lda_probabilities = lda_model.predict_proba(X_test)[:, 1]
    # # 检查并处理 NaN 值
    # lda_nan_indices = np.isnan(lda_probabilities)
    # print(lda_nan_indices)
    # lda_probabilities[lda_nan_indices] = 0  # 将 NaN 值替换为其他值，如 0
    # print("END training with LDA")

    y_test_binary = y_test #y_test_binary[:, 0, 1]
    # print(y_test_binary.shape)
    # print(lda_probabilities.shape)

    # Compute the false positive rate (fpr) and true positive rate (tpr) for each model
    textcnn_fpr, textcnn_tpr, _ = roc_curve(y_test, textcnn_probabilities)
    # xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probabilities)
    # rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probabilities)
    # gru_fpr, gru_tpr, _ = roc_curve(y_test_binary, gru_probabilities)
    # lda_fpr, lda_tpr, _ = roc_curve(y_test_binary, lda_probabilities)
    # # Compute the AUC scores for each model
    textcnn_auc = auc(textcnn_fpr, textcnn_tpr)
    # xgb_auc = auc(xgb_fpr, xgb_tpr)
    # rf_auc = auc(rf_fpr, rf_tpr)
    # lda_auc = auc(lda_fpr, lda_tpr)
    # gru_auc = auc(gru_fpr, gru_tpr)
    # # Append the AUC scores to the lists
    textcnn_aucs.append(textcnn_auc)
    # xgb_aucs.append(xgb_auc)
    # rf_aucs.append(rf_auc)
    # lda_aucs.append(lda_auc)
    # gru_aucs.append(gru_auc)

    # lda_predict = lda_model.predict(X_test)
    # lda_acc = accuracy_score(y_pred=lda_predict, y_true=y_test)
    # lda_accs.append(lda_acc)

    threshold = 0.8  # 阈值
    # 将概率值转换为类别标签
    # binary_predictions = [1 if prob >= threshold else 0 for prob in gru_probabilities]
    # gru_acc = accuracy_score(y_pred=binary_predictions, y_true=y_test)
    # gru_accs.append(gru_acc)

    textcnn_binary_predictions = [1 if prob >= threshold else 0 for prob in textcnn_probabilities]

    textcnn_acc = accuracy_score(y_pred=textcnn_binary_predictions, y_true=y_test)
    textcnn_accs.append(textcnn_acc)
# Compute the mean AUC scores
textcnn_mean_auc = np.mean(textcnn_aucs)
# xgb_mean_auc = np.mean(xgb_aucs)
# rf_mean_auc = np.mean(rf_aucs)
# lda_mean_auc = np.mean(lda_aucs)
# gru_mean_auc = np.mean(gru_aucs)

fig = plt.figure()
# Plot the mean AUC curves
# plt.plot(lda_fpr, lda_tpr, label=f'LDA (AUC = {lda_mean_auc:.2f})')
# plt.plot(gru_fpr, gru_tpr, label=f'GRU (AUC = {gru_mean_auc:.2f})')
plt.plot(textcnn_fpr, textcnn_tpr, label=f'TextCNN (AUC = {textcnn_mean_auc:.2f})')
# plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_mean_auc:.2f})')
# plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_mean_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k', label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

# Set labels and title
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.tick_params(axis='both', which='major', labelsize=14)  # 设置刻度标签大小
plt.legend(loc="lower right", fontsize=12)  # 设置图例字体大小
plt.grid(alpha=0.3)  # 添加网格线
# plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()
fig = plt.figure()
# fig.suptitle('Classifier Comparison')
ax = fig.add_subplot(111)
ax.yaxis.grid(True)
# plt.boxplot([cnn_accs,textcnn_accs,lstm_accs,bilstm_accs,cnnlstm_accs,xgb_accs,lgb_accs,svm_accs,lr_accs,dt_accs,nb_accs,knn_accs,lda_accs],
#             patch_artist=True, vert=True, whis=True, showbox=True)
plt.boxplot([textcnn_accs],
            patch_artist=True, vert=True, whis=True, showbox=True)
# ax.set_xticklabels(['textcnn','xgb','lgb','svm','lr','dt','knn','lda'], fontsize=12)
ax.set_xticklabels(['textcnn'], fontsize=12)
plt.xlabel('Classifiers', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()