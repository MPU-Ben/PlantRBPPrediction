import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, f1_score, matthews_corrcoef, \
    confusion_matrix, precision_score
import tensorflow as tf
def load_and_predict(model_path, new_data):
    model = tf.keras.models.load_model(model_path)
    X = new_data.iloc[:, :-1].values
    # y = new_data.iloc[:, -1].values
    scaler = joblib.load('scaler.pkl')
    new_data=scaler.transform(X)
    reshaped_new_data = np.expand_dims(new_data, axis=-1)
    predictions = model.predict(reshaped_new_data)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes

IndependenceTestingData = pd.read_csv('../Files/IndependenceTestingDataset_0406_NO_RF_GBDT_XGB.csv')

predictions = load_and_predict('final_textcnn_model.keras', IndependenceTestingData)

# 输出评价指标
# 假设 true_labels 是新数据的真实标签
true_labels = IndependenceTestingData.iloc[:, -1].values
acc = accuracy_score(true_labels, predictions) * 100
f1 = f1_score(true_labels, predictions, average='weighted') * 100
mcc = matthews_corrcoef(true_labels, predictions) * 100
precision = precision_score(true_labels, predictions, average='weighted') * 100  # 计算精确率
cm = confusion_matrix(true_labels, predictions)
tn, fp, fn, tp = cm.ravel()
sn = round(tp / (tp + fn) * 100, 2)
sp = round(tn / (tn + fp) * 100, 2)
print(f'Evaluate on Independent dataset')
print(f'Accuracy: {acc:.2f}%')
print(f'F1 Score: {f1:.2f}%')
print(f'MCC: {mcc:.2f}%')
print(f'Precision: {precision:.2f}%')
print(f'SN: {sn:.2f}%')
print(f'SP: {sp:.2f}%')
# 绘制 ROC 曲线
if predictions.ndim == 1:  # 如果是1维数组，说明是二分类的概率
    fpr, tpr, _ = roc_curve(true_labels, predictions)  # 使用正类的概率
else:
    fpr, tpr, _ = roc_curve(true_labels, predictions[:, 1])  # 取正类的预测概率

roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # 对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('roc_curve2.png', dpi=300)  # 保存 ROC 曲线图
plt.show()


# 绘制 PR 曲线
precision, recall, _ = precision_recall_curve(true_labels, predictions)  # 取正类的预测概率
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.savefig('pr_curve2.png', dpi=300)  # 保存 PR 曲线图
plt.show()