import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn import svm
from keras.src.utils import np_utils, to_categorical
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import models.create_textcnn_model
import models.create_cnn_model
import models.create_lstm_model
import models.create_bilstm_model
import models.create_cnnlstm_model
import models.create_gru_model
import models.create_attention_gru_model
import models.create_textrnn_att_model
import lightgbm as lgb
from tensorflow.keras.callbacks import Callback
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
# import gcforest
from gcforest.gcforest import GCForest
# from GCForest import gcforest
import joblib
def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100 ##最大层数
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2 ##类别数
    ##选择级联森林的基模型
    ca_config["estimators"] = []
    ca_config["estimators"].append(
            {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
             "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 1, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config

class PrintBatchAccuracyCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        accuracy = logs['accuracy']
        loss =  logs['loss']
        print(f"Batch {batch+1} - Accuracy: {accuracy:.4f} Loss:{loss:.4f}")

# create callback function
print_batch_accuracy_callback = PrintBatchAccuracyCallback()

# Assuming you have the dataset in a CSV file
data = pd.read_csv('../files/optimumAllTrainDataset.csv')

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
feature_dim = X.shape[1]
num_classes = len(set(y))  # 标签类别数量
num_epochs = 5
num_batch_size = 64
threshold = 0.5  # 阈值
# y = np_utils.to_categorical(y, num_classes=num_classes)
# 定义基本模型
base_models = [
    # RandomForestClassifier(n_estimators=100, random_state=42),
    # # LogisticRegression(random_state=42)
    # svm.SVC(probability=True),
    # LGBMClassifier(n_estimators= 200),
    # XGBClassifier(n_estimators= 100)
    RandomForestClassifier(n_estimators=100, random_state=42),
    RandomForestClassifier(n_estimators=200, random_state=42),
    RandomForestClassifier(n_estimators=300, random_state=42)
]
# 定义元模型
meta_model = LogisticRegression()
# 定义Stacking Ensemble模型
# stacking_model = StackingClassifier(classifiers=base_models, meta_classifier=meta_model)
stacking_model = StackingCVClassifier(classifiers=base_models, meta_classifier=LGBMClassifier(random_state=42))
attention_gru_model = models.create_attention_gru_model.gru_attention_model(feature_dim)
# from deepforest import CascadeForestClassifier
# gcforest_model = GCForest(get_toy_config()) ##构建模型gcforest_model
# gcforest_model = gcforest.gcforest.
cnn_model = models.create_cnn_model.cnn_model(feature_dim)
textcnn_model = models.create_textcnn_model.textcnn_model(feature_dim)
textrnn_att_model = models.create_textrnn_att_model.text_rnn_attention_model(feature_dim)
lstm_model = models.create_lstm_model.lstm_model(feature_dim)
bilstm_model = models.create_bilstm_model.bilstm_model(feature_dim)
cnnlstm_model = models.create_cnnlstm_model.cnnlstm_model(feature_dim)
gru_model = models.create_gru_model.gru_model(feature_dim)
# XGBoost model
xgb_model = XGBClassifier(colsample_bytree=1.0, learning_rate=0.1, max_depth=5,
        n_estimators= 500, reg_alpha=0, reg_lambda=0.1, subsample=0.8,tree_method='approx')
lgb_model = lgb.LGBMClassifier(colsample_bytree=0.8, learning_rate= 0.1, n_estimators= 200, num_leaves=31, subsample=0.8,verbose=0)
svm_model = svm.SVC(probability=True)
lr_model = LogisticRegression(penalty='l2', C=1, max_iter=100, solver='newton-cg')
dt_model = DecisionTreeClassifier(criterion='gini', max_depth = 5, min_samples_leaf= 2, min_samples_split= 2)
nb_model = GaussianNB(priors=None, var_smoothing= 1e-09)
knn_model = KNeighborsClassifier(algorithm='auto',n_neighbors=3,weights='distance')
lda_model = LinearDiscriminantAnalysis(n_components = None, shrinkage = None, solver = 'lsqr')

# aucs
attention_gru_aucs = []
stacking_model_aucs = []
cnn_aucs = []
textcnn_aucs = []
textrnn_att_aucs = []
lstm_aucs = []
bilstm_aucs = []
cnnlstm_aucs = []
gru_aucs = []
gcforest_aucs = []
xgb_aucs = []
lgb_aucs = []
svm_aucs = []
lr_aucs = []
dt_aucs = []
nb_aucs = []
knn_aucs = []
lda_aucs = []
# accs
attention_gru_accs = []
cnn_accs=[]
textcnn_accs=[]
textrnn_att_accs = []
lstm_accs=[]
bilstm_accs=[]
cnnlstm_accs=[]
gru_accs = []
stacking_model_accs = []
gcforest_accs = []
xgb_accs = []
lgb_accs = []
svm_accs = []
lr_accs = []
dt_accs = []
nb_accs = []
knn_accs = []
lda_accs = []

avg_svm_prop = []
fold_count = 0
# Perform 5-fold cross-validation
# F = open('../Files/evaluationResults.txt', 'w')
#
# F.write('Evaluation Scale:'+'\n')
# F.write('0.0% <=Accuracy<= 100.0%'+'\n')
# F.write('0.0 <=auROC<= 1.0'+'\n')
# F.write('0.0 <=auPR<= 1.0'+'\n')  # average_Precision
# F.write('0.0 <=F1_Score<= 1.0'+'\n')
# F.write('-1.0 <=MCC<= 1.0'+'\n')
# F.write('0.0%<=Sensitivity<= 100.0%'+'\n')
# F.write('0.0%<=Specificity<= 100.0%'+'\n')
#############进行归一化################
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(X)
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# for train_index, test_index in kfold.split(X):
# X_train, X_test = X[train_index], X[test_index]
# y_train, y_test = y[train_index], y[test_index]
# fold_count = fold_count + 1
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,shuffle=True)
print("current training fold number:",fold_count)
print("y_train.shape:",y_train.shape)
# print("y_train_data:",y_train[:])
print("X_train.shape[0]:,",X_train.shape[0])
reshaped_X_train = np.zeros((X_train.shape[0], X_train.shape[1], 4))
reshaped_X_train[:, :, 0] = X_train[:, :]
# reshaped_X_train = X_train.reshape((sample_num, feature_dim, 1))
print("reshaped_X_train",reshaped_X_train.shape)
############################################
reshaped_X_test = np.zeros((X_test.shape[0], X_test.shape[1], 4))
reshaped_X_test[:, :, 0] = X_test[:, :]
# reshaped_X_train = X_train.reshape((sample_num, feature_dim, 1))
print("reshaped_X_test",reshaped_X_test.shape)
###########################################
y_train_encoded = to_categorical(y_train,num_classes)

# print("start training with ATTENTION CNN")
# att_gru_history = attention_gru_model.fit(X_train, y_train_encoded, epochs=num_epochs, batch_size = num_batch_size, verbose=0, callbacks=[print_batch_accuracy_callback])
# att_gru_train_loss = att_gru_history.history['loss']
# print("train_loss",att_gru_train_loss)
# # val_loss = history.history['val_loss']
# att_gru_train_accuracy = att_gru_history.history['accuracy']
# print("train_accuracy:",att_gru_train_accuracy)
# att_gru_probabilities = attention_gru_model.predict(X_test)[:, 1]
# print("END training with ATTENTION CNN")
#
# print("start training with CNN")
# cnn_history = cnn_model.fit(reshaped_X_train, y_train_encoded, epochs=num_epochs,batch_size = num_batch_size, verbose=0, callbacks=[print_batch_accuracy_callback])
# cnn_train_loss = cnn_history.history['loss']
# print("train_loss",cnn_train_loss)
# # val_loss = history.history['val_loss']
# cnn_train_accuracy = cnn_history.history['accuracy']
# print("train_accuracy:",cnn_train_accuracy)
# cnn_probabilities = cnn_model.predict(reshaped_X_test)
# print("END training with CNN")
#
# # TextCNN model training and prediction
print("start training with TextCNN")
history = textcnn_model.fit(reshaped_X_train, y_train, epochs=num_epochs, batch_size = num_batch_size, verbose=0,callbacks=[print_batch_accuracy_callback])#callbacks=[print_batch_accuracy_callback]
train_key = history.history.keys()
# print(train_key)
train_loss = history.history['loss']
print("train_loss",train_loss)
# val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
print("train_accuracy:",train_accuracy)
textcnn_probabilities = textcnn_model.predict(reshaped_X_test)
text_result = textcnn_model.evaluate(reshaped_X_test,y_test)
print(text_result)
print("END training with TextCNN")
#
print("start training with TextRNN Attention")
history = textrnn_att_model.fit(reshaped_X_train, y_train_encoded, epochs=num_epochs, batch_size = num_batch_size, verbose=0,callbacks=[print_batch_accuracy_callback])#callbacks=[print_batch_accuracy_callback]
train_key = history.history.keys()
# print(train_key)
train_loss = history.history['loss']
print("train_loss",train_loss)
# val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
print("train_accuracy:",train_accuracy)
textrnn_att_probabilities = textrnn_att_model.predict(reshaped_X_test)
textrnn_att_result = textrnn_att_model.evaluate(reshaped_X_test,y_train_encoded)
print(textrnn_att_result)
print("END training with TextRNN Attention")
#
print("start training with LSTM")
lstm_history = lstm_model.fit(reshaped_X_train, y_train_encoded, epochs=num_epochs, batch_size=num_batch_size, verbose=0,callbacks=[print_batch_accuracy_callback])
lstm_train_loss = lstm_history.history['loss']
print("train_loss",lstm_train_loss)
# val_loss = history.history['val_loss']
lstm_train_accuracy = lstm_history.history['accuracy']
print("train_accuracy:",lstm_train_accuracy)
lstm_probabilities = lstm_model.predict(reshaped_X_test)
print("END training with LSTM")
#
print("start training with BiLSTM")
bilstm_history = bilstm_model.fit(reshaped_X_train, y_train_encoded, epochs=num_epochs, batch_size=num_batch_size, verbose=0,callbacks=[print_batch_accuracy_callback])
bilstm_train_loss = bilstm_history.history['loss']
print("train_loss",bilstm_train_loss)
# val_loss = history.history['val_loss']
bilstm_train_accuracy = bilstm_history.history['accuracy']
print("train_accuracy:",bilstm_train_accuracy)
bilstm_probabilities = bilstm_model.predict(reshaped_X_test)
print("END training with BiLSTM")
#
# print("start training with CNN_LSTM")
# cnnlstm_history = cnnlstm_model.fit(reshaped_X_train, y_train_encoded, epochs=num_epochs, batch_size=64, verbose=0)
# cnnlstm_train_loss = cnnlstm_history.history['loss']
# print("train_loss", cnnlstm_train_loss)
# cnnlstm_train_accuracy = cnnlstm_history.history['accuracy']
# print("train_accuracy:", cnnlstm_train_accuracy)
# cnnlstm_probabilities = cnnlstm_model.predict(X_test)
# print("END training with CNN_LSTM")
#
# stacking model training and prediction
# print("start training with gcforest model ")
# gcforest_model.fit_transform(X_train, y_train)
# gcforest_probabilities = gcforest_model.predict_proba(X_test)[:, 1]
# gcforest_nan_indices = np.isnan(gcforest_probabilities)
# print(gcforest_nan_indices)
# gcforest_probabilities[gcforest_nan_indices]=0
# print("END training with gcforest model")

# stacking model training and prediction
# print("start training with stacking model ")
# stacking_model.fit(X_train, y_train)
# stacking_model_probabilities = stacking_model.predict_proba(X_test)[:, 1]
# lgb_nan_indices = np.isnan(stacking_model_probabilities)
# print(lgb_nan_indices)
# stacking_model_probabilities[lgb_nan_indices]=0
# print("END training with stacking model")
#
# XGBoost model training and prediction
print("start training with XGB")
xgb_model.fit(X_train, y_train)
xgb_probabilities = xgb_model.predict_proba(X_test)[:, 1]
xgb_nan_indices = np.isnan(xgb_probabilities)
# print(xgb_nan_indices)
xgb_probabilities[xgb_nan_indices]=0
print("END training with XGB")
#
# # lightgbm model training and prediction
# print("start training with lightgbm")
# lgb_model.fit(X_train, y_train)
# lgb_probabilities = lgb_model.predict_proba(X_test)[:, 1]
# lgb_nan_indices = np.isnan(lgb_probabilities)
# print(lgb_nan_indices)
# lgb_probabilities[lgb_nan_indices]=0
# print("END training with lightgbm")

#
# svm model training and prediction
# print("start training with svm")
# svm_model.fit(X_train, y_train)
# svm_probabilities = svm_model.predict_proba(X_test)[:, 1]
# print("svm_probabilities shape:",np.array(svm_probabilities).shape)
# avg_svm_prop.append(svm_probabilities)
#
# svm_nan_indices = np.isnan(svm_probabilities)
# print(svm_nan_indices)
# svm_probabilities[svm_nan_indices]=0
# print("END training with svm")
#
# # logistregression model training and prediction
# print("start training with logistregression")
# lr_model.fit(X_train, y_train)
# lr_probabilities = lr_model.predict_proba(X_test)[:, 1]
# lr_nan_indices = np.isnan(lr_probabilities)
# print(lr_nan_indices)
# lr_probabilities[lr_nan_indices]=0
# print("END training with logistregression")
# #
# # # decisiontree model training and prediction
# print("start training with decisiontree")
# dt_model.fit(X_train, y_train)
# dt_probabilities = dt_model.predict_proba(X_test)[:, 1]
# dt_nan_indices = np.isnan(dt_probabilities)
# print(dt_nan_indices)
# dt_probabilities[dt_nan_indices]=0
# print("END training with decisiontree")
# #
# # # NB model training and prediction
# print("start training with NB")
# nb_model.fit(X_train, y_train)
# nb_probabilities = nb_model.predict_proba(X_test)[:, 1]
# nb_nan_indices = np.isnan(nb_probabilities)
# print(nb_nan_indices)
# nb_probabilities[nb_nan_indices]=0
# print("END training with NB")
#
# # # KNN model training and prediction
# print("start training with KNN")
# knn_model.fit(X_train, y_train)
# knn_probabilities = knn_model.predict_proba(X_test)[:, 1]
# knn_nan_indices = np.isnan(knn_probabilities)
# print(knn_nan_indices)
# knn_probabilities[knn_nan_indices]=0
# print("END training with KNN")
# #
# # # LDA model training and prediction
# print("start training with LDA")
# lda_model.fit(X_train, y_train)
# lda_probabilities = lda_model.predict_proba(X_test)[:, 1]
# # 检查并处理 NaN 值
# lda_nan_indices = np.isnan(lda_probabilities)
# print(lda_nan_indices)
# lda_probabilities[lda_nan_indices] = 0  # 将 NaN 值替换为其他值，如 0
# print("END training with LDA")
# Compute the false positive rate (fpr) and true positive rate (tpr) for each model
# Convert labels to binary format
# y_test_binary = to_categorical(y_test, num_classes=num_classes)
# Select the first class as the positive class
# print("start training with GRU")
# gru_model.fit(reshaped_X_train, y_train_encoded, epochs=num_epochs, batch_size = num_batch_size, verbose=0,)
# gru_probabilities = gru_model.predict(X_test)[:, 1]
# # 检查并处理 NaN 值
# gru_nan_indices = np.isnan(gru_probabilities)
# print(gru_nan_indices)
# gru_probabilities[gru_nan_indices] = 0  # 将 NaN 值替换为其他值，如 0
# print("END training with GRU")

y_test_binary = y_test #y_test_binary[:, 0, 1]
print(y_test_binary.shape)

# Y_pred = [np.argmax(y) for y in cnn_probabilities] # 取出y中元素最大值所对应的索引
# Compute the false positive rate (fpr) and true positive rate (tpr) for each model
# cnn_fpr, cnn_tpr, _ = roc_curve(y_test_binary, cnn_probabilities[:, 0])
textcnn_fpr, textcnn_tpr, _ = roc_curve(y_test_binary, textcnn_probabilities[:, 0])
textrnn_att_fpr, textrnn_att_tpr, _ = roc_curve(y_test_binary, textrnn_att_probabilities[:, 0])
# attention_gru_fpr, attention_gru_tpr = roc_curve(y_test_binary,att_gru_probabilities)
lstm_fpr, lstm_tpr, _ = roc_curve(y_test_binary, lstm_probabilities[:, 0])
bilstm_fpr, bilstm_tpr, _ = roc_curve(y_test_binary, bilstm_probabilities[:, 0])
# cnnlstm_fpr, cnnlstm_tpr, _ = roc_curve(y_test_binary, cnnlstm_probabilities[:, 0])
# gru_fpr, gru_tpr, _ = roc_curve(y_test_binary, gru_probabilities)
# gcforest_fpr, gcforest_tpr, _ = roc_curve(y_test_binary, gcforest_probabilities)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test_binary, xgb_probabilities)


# Compute the AUC scores for each model
# cnn_auc = auc(cnn_fpr, cnn_tpr)
textcnn_auc = auc(textcnn_fpr, textcnn_tpr)
textrnn_at_auc = auc(textrnn_att_fpr,textrnn_att_tpr)
# attention_gru_auc = auc(attention_gru_fpr,attention_gru_tpr)
lstm_auc = auc(lstm_fpr, lstm_tpr)
bilstm_auc = auc(bilstm_fpr, bilstm_tpr)
# cnnlstm_auc = auc(cnnlstm_fpr, cnnlstm_tpr)
# gru_auc = auc(gru_fpr, gru_tpr)
# gcforest_auc = auc(gcforest_fpr,gcforest_tpr)
xgb_auc = auc(xgb_fpr, xgb_tpr)


# Append the AUC scores to the lists
# cnn_aucs.append(cnn_auc)
textcnn_aucs.append(textcnn_auc)
textrnn_att_aucs.append(textrnn_at_auc)
# attention_gru_aucs.append(attention_gru_auc)
lstm_aucs.append(lstm_auc)
bilstm_aucs.append(bilstm_auc)
# cnnlstm_aucs.append(cnnlstm_auc)
# gru_aucs.append(gru_auc)
# gcforest_aucs(gcforest_auc)
xgb_aucs.append(xgb_auc)
# lgb_aucs.append(lgb_auc)
# stacking_model_aucs.append(stacking_model_auc)


# cnn_acc = accuracy_score(y_pred=np.argmax(cnn_probabilities,axis=1), y_true=y_test)
# cnn_accs.append(cnn_acc)

textcnn_acc = accuracy_score(y_pred=np.argmax(textcnn_probabilities,axis=1), y_true=y_test)
textcnn_accs.append(textcnn_acc)
#
textrnn_att_acc = accuracy_score(y_pred=np.argmax(textrnn_att_probabilities,axis=1), y_true=y_test)
textrnn_att_accs.append(textrnn_att_acc)
# att_gru_binary_predictions = [1 if prob >= threshold else 0 for prob in att_gru_probabilities]
# att_gru_acc = accuracy_score(y_pred=att_gru_binary_predictions, y_true=y_test)
# attention_gru_accs.append(att_gru_acc)
#
lstm_acc = accuracy_score(y_pred=np.argmax(lstm_probabilities,axis=1), y_true=y_test)
lstm_accs.append(lstm_acc)
#
bilstm_acc = accuracy_score(y_pred=np.argmax(bilstm_probabilities,axis=1), y_true=y_test)
bilstm_accs.append(bilstm_acc)
#
# cnnlstm_acc = accuracy_score(y_pred=np.argmax(cnnlstm_probabilities,axis=1), y_true=y_test)
# cnn_accs.append(cnnlstm_acc)

# 将概率值转换为类别标签
# binary_predictions = [1 if prob >= threshold else 0 for prob in gru_probabilities]
# gru_acc = accuracy_score(y_pred=binary_predictions, y_true=y_test)
# gru_accs.append(gru_acc)
# gcforest_predict = gcforest_model.predict(X_test)
# gcfoest_acc = accuracy_score(y_pred=gcforest_predict, y_true=y_test)
# gcforest_accs.append(gcfoest_acc)

xgb_predict = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_pred=xgb_predict, y_true=y_test)
xgb_accs.append(xgb_acc)
#
# lgb_predict = lgb_model.predict(X_test)
# lgb_acc = accuracy_score(y_pred=lgb_predict, y_true=y_test)
# lgb_accs.append(lgb_acc)
#
# stacking_model_predict = stacking_model.predict(X_test)
# stacking_model_acc = accuracy_score(y_pred=stacking_model_predict, y_true=y_test)
# stacking_model_accs.append(stacking_model_acc)
#
# svm_predict = svm_model.predict(X_test)
# svm_acc = accuracy_score(y_pred=svm_predict, y_true=y_test)
# svm_accs.append(svm_acc)
# #
# lr_predict = lr_model.predict(X_test)
# lr_acc = accuracy_score(y_pred=lr_predict, y_true=y_test)
# lr_accs.append(lr_acc)
#
# dt_predict = dt_model.predict(X_test)
# dt_acc = accuracy_score(y_pred=dt_predict, y_true=y_test)
# dt_accs.append(dt_acc)
#
# nb_predict = nb_model.predict(X_test)
# nb_acc = accuracy_score(y_pred=nb_predict, y_true=y_test)
# nb_accs.append(nb_acc)
#
# knn_predict = knn_model.predict(X_test)
# knn_acc = accuracy_score(y_pred=knn_predict, y_true=y_test)
# knn_accs.append(knn_acc)
#
# lda_predict = lda_model.predict(X_test)
# lda_acc = accuracy_score(y_pred=lda_predict, y_true=y_test)
# lda_accs.append(lda_acc)

# Compute the mean AUC scores
cnn_mean_auc = np.mean(cnn_aucs)
textcnn_mean_auc = np.mean(textcnn_aucs)
textrnn_att_mean_auc = np.mean(textrnn_att_aucs)
# att_gru_mean_auc = np.mean(attention_gru_aucs)
lstm_mean_auc = np.mean(lstm_aucs)
bilstm_mean_auc = np.mean(bilstm_aucs)
# cnnlstm_mean_auc = np.mean(cnnlstm_aucs)
# gru_mean_auc = np.mean(gru_aucs)
# gcforest_mean_auc = np.mean(gcforest_aucs)
xgb_mean_auc = np.mean(xgb_aucs)
# lgb_mean_auc = np.mean(lgb_aucs)
# stacking_model_mean_auc = np.mean(stacking_model_aucs)
# svm_mean_auc = np.mean(svm_aucs)
# lr_mean_auc = np.mean(lr_aucs)
# dt_mean_auc = np.mean(dt_aucs)
# nb_mean_auc = np.mean(nb_aucs)
# knn_mean_auc = np.mean(knn_aucs)
# lda_mean_auc = np.mean(lda_aucs)
#
# print("avg_svm_prop",np.array(avg_svm_prop).shape)
fig = plt.figure()
# Plot the mean AUC curves
# plt.plot(cnn_fpr, cnn_tpr, label=f'CNN (AUC = {cnn_mean_auc:.2f})')
plt.plot(textcnn_fpr, textcnn_tpr, label=f'TextCNN (AUC = {textcnn_mean_auc:.2f})')
plt.plot(textrnn_att_fpr, textrnn_att_tpr, label=f'TextRNN ATT (AUC = {textrnn_att_mean_auc:.2f})')
# plt.plot(attention_gru_fpr, attention_gru_tpr, label=f'ATT_GRU (AUC = {textcnn_mean_auc:.2f})')
plt.plot(lstm_fpr, lstm_tpr, label=f'LSTM (AUC = {lstm_mean_auc:.2f})')
plt.plot(bilstm_fpr, bilstm_tpr, label=f'BiLSTM (AUC = {bilstm_mean_auc:.2f})')
# plt.plot(cnnlstm_fpr, cnnlstm_tpr, label=f'CNN_LSTM (AUC = {cnnlstm_mean_auc:.2f})')
# plt.plot(gru_fpr,gru_tpr, label=f'GRU (AUC = {gru_mean_auc:.2f})')
# plt.plot(gcforest_fpr, gcforest_tpr, label=f'gcForest (AUC = {gcforest_mean_auc:.2f})')
plt.plot(xgb_fpr, xgb_tpr, label=f'LMFE (AUC = {xgb_mean_auc:.2f})')
# plt.plot(lgb_fpr, lgb_tpr, label=f'LightGBM (AUC = {lgb_mean_auc:.2f})')
# plt.plot(stacking_model_fpr, stacking_model_tpr, label=f'Stacking (AUC = {stacking_model_mean_auc:.2f})')
# plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {svm_mean_auc:.2f})')
# plt.plot(lr_fpr, lr_tpr, label=f'LR (AUC = {lr_mean_auc:.2f})')
# plt.plot(dt_fpr, dt_tpr, label=f'DT (AUC = {dt_mean_auc:.2f})')
# plt.plot(nb_fpr, nb_tpr, label=f'NB (AUC = {nb_mean_auc:.2f})')
# plt.plot(knn_fpr, knn_tpr, label=f'KNN (AUC = {knn_mean_auc:.2f})')
# plt.plot(lda_fpr, lda_tpr, label=f'LDA (AUC = {lda_mean_auc:.2f})')
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
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()
# Show the plot
fig = plt.figure()
# fig.suptitle('Classifier Comparison')
ax = fig.add_subplot(111)
ax.yaxis.grid(True)
# plt.boxplot([cnn_accs,textcnn_accs,lstm_accs,bilstm_accs,cnnlstm_accs,xgb_accs,lgb_accs,svm_accs,lr_accs,dt_accs,nb_accs,knn_accs,lda_accs],
#             patch_artist=True, vert=True, whis=True, showbox=True)
# plt.boxplot([svm_accs,lr_accs,dt_accs,nb_accs,knn_accs,lstm_accs,xgb_accs,lgb_accs],
plt.boxplot([textcnn_accs,textrnn_att_accs,lstm_accs,bilstm_accs,xgb_accs],
            patch_artist=True, vert=True, whis=True, showbox=True)
# ax.set_xticklabels(['textcnn','xgb','lgb','svm','lr','dt','knn','lda'], fontsize=12)
ax.set_xticklabels(['textcnn','textrnn att','lstm','bilstm','LMFE'], fontsize=12)
plt.xlabel('Classifiers', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()
