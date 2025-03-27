import numpy as np

import pandas as pd
import seaborn as sns
import tensorflow as tf
# from tensorflow import keras
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, matthews_corrcoef, confusion_matrix, f1_score
from tensorflow.python.keras.utils.np_utils import to_categorical

import models.create_textcnn_model
tf.config.experimental_run_functions_eagerly(False)
# data = pd.read_csv('../files/optimumAllTrainingDataset_F0.csv')
data = pd.read_csv('../files/optimumAllTrainingDataset.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# from sklearn.preprocessing import StandardScaler
# scale = StandardScaler()
# X = scale.fit_transform(X)
# X_train,X_val,y_train,y_val=train_test_split(X,y,shuffle=True,test_size=0.5,stratify=y)
# testing_data = pd.read_csv('../files/optimumIndependenceTestingDataset.csv')
testing_data = pd.read_csv('../files/optimumAllTestDataset.csv')
X_independent_test = data.iloc[:, :-1].values
y_independent_test = data.iloc[:, -1].values
num_classes = len(set(y))  # 标签类别数量
feature_dim = X.shape[1]
num_epochs=50
num_batch_size = 32
textcnn_model = models.create_textcnn_model.textcnn_model(feature_dim)
textcnn_aucs, textcnn_accs, textcnn_mccs, textcnn_f1s, textcnn_sns, textcnn_sps = ([] for _ in range(6))

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    reshaped_X_train = np.expand_dims(X_train, axis=-1)  # **修改：将数据从二维扩展为三维**
    reshaped_X_test = np.expand_dims(X_test, axis=-1)  # **修改：将数据从二维扩展为三维**

    y_train_encoded = to_categorical(y_train, num_classes)
    y_test_encoded = to_categorical(y_test, num_classes)

    history = textcnn_model.fit(reshaped_X_train, y_train_encoded, epochs=num_epochs,
                                batch_size=num_batch_size, verbose=0)
    predictions = textcnn_model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test_encoded, axis=1)

    acc = round(accuracy_score(true_classes, predicted_classes) * 100,2)
    fpr, tpr, _ = roc_curve(true_classes, predictions[:, 1])  # 获取正类的概率
    auc_score = round(auc(fpr, tpr) *100,2)
    mcc = round(matthews_corrcoef(true_classes, predicted_classes) * 100,2)  # 计算 MCC
    f1 = round(f1_score(true_classes, predicted_classes, average='weighted') * 100,2)
    cm = confusion_matrix(true_classes, predicted_classes)

    tn, fp, fn, tp = cm.ravel()
    sn = round(tp / (tp + fn) * 100,2)
    sp = round(tn / (tn + fp) * 100,2)

    textcnn_accs.append(acc)
    textcnn_aucs.append(auc_score)
    textcnn_mccs.append(mcc)  # 存储 MCC
    textcnn_f1s.append(f1)
    textcnn_sns.append(sn)
    textcnn_sps.append(sp)
# 计算平均 AUC、准确率和 MCC
# 计算平均值
mean_metrics = {
    'Accuracy': np.mean(textcnn_accs),
    'AUC': np.mean(textcnn_aucs),
    'MCC': np.mean(textcnn_mccs),
    'F1_SCORE': np.mean(textcnn_f1s),
    'SN': np.mean(textcnn_sns),
    'SP': np.mean(textcnn_sps)
}

# 打印结果
print("\nAverage Metrics Across 5 Folds:")
for metric, value in mean_metrics.items():
    print(f"Mean {metric}: {value:.4f}")
# 可视化
# 将指标整理成DataFrame用于可视化
metrics_df = pd.DataFrame({
    'ACC': textcnn_accs,
    'AUC': textcnn_aucs,
    'MCC': textcnn_mccs,
    'F1score': textcnn_f1s,
    'SN': textcnn_sns,
    'SP': textcnn_sps
})

import matplotlib
matplotlib.use('TkAgg')  # 更改后端
import matplotlib.pyplot as plt
# 设置字体
matplotlib.rc('font', family='Times New Roman')

plt.style.use('seaborn-v0_8-whitegrid')  # 美化样式
# 1. 箱线图：展示所有指标的分布
plt.figure(figsize=(12, 6))
sns.boxplot(data=metrics_df, palette='Set2')
# 添加中位数的文本
for i in range(len(metrics_df.columns)):
    median = metrics_df.iloc[:, i].median()  # 计算中位数并转换为百分比
    plt.text(i, median, f'{median:.1f}', color='black', ha='center', va='bottom')
# plt.title('Metrics Distribution Across 5-Folds ', fontsize=14)
plt.ylabel('Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.ylim(95, 102)
plt.tight_layout()
plt.savefig('boxplot_metrics.png', dpi=300)
plt.show()

# # 将 metrics_df 中的指标转换为适合 plt.boxplot 的格式
# data_to_plot = [
#     metrics_df['ACC'] * 100,  # 转换为百分比
#     metrics_df['AUC'] * 100,
#     metrics_df['MCC'] * 100,
#     metrics_df['F1score'] * 100,
#     metrics_df['SN'] * 100,
#     metrics_df['SP'] * 100
# ]
#
# # 创建箱线图
# plt.figure(figsize=(12, 6))
# plt.boxplot(data_to_plot, patch_artist=True, boxprops=dict(facecolor='lightblue'))
#
# plt.xticks(
#     ticks=np.arange(1, len(data_to_plot) + 1),
#     labels=['ACC', 'AUC', 'MCC', 'F1score', 'SN', 'SP'],
#     rotation=45
# )
#
# plt.title('Metrics Distribution Across 5-Folds', fontsize=14)
# plt.ylabel('Score (0-100)', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)
#
# # 设置 y 轴范围
# plt.ylim(80, 100)

# 调整布局并保存图像
# plt.tight_layout()
# plt.savefig('boxplot_metrics_plt.png', dpi=300)
# plt.show()
#
# import plotnine as p9
# from plotnine import ggplot, aes, geom_boxplot, labs
# plotnine_plot = (
#     p9.ggplot(metrics_df, p9.aes(x='variable', y='value')) +
#     p9.geom_boxplot(fill='lightblue') +
#     p9.labs(title='Metrics Distribution Across 5-Folds', y='Score') +
#     p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1),
#              text=p9.element_text(family='Times New Roman'))
# )
#
# # 显示 Plotnine 图形
# print(plotnine_plot)