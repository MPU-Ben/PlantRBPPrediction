import time
import psutil
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_curve, auc, accuracy_score,
                             matthews_corrcoef, confusion_matrix, f1_score, precision_recall_curve)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.utils.np_utils import to_categorical
import Program.models.create_textcnn_model
import matplotlib
import matplotlib.pyplot as plt
fold = 10
# 设置 matplotlib 后端和字体
matplotlib.use('TkAgg')
matplotlib.rc('font', family='Times New Roman')
plt.style.use('seaborn-v0_8-whitegrid')
# 定义训练模型的函数
# 获取当前进程的 PID
pid = psutil.Process().pid
p = psutil.Process(pid)
cpu_before = p.cpu_percent(interval=None)  # 初始化 CPU 使用率
mem_before = p.memory_info().rss / (1024 * 1024)  # 转换为 MB
def training_model(dataset):
    print('curent dataset:',dataset)
    data = pd.read_csv(dataset)
    # plot_cor(data)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    num_classes = len(set(y))
    feature_dim = X.shape[1]
    num_epochs = 50
    num_batch_size = 32
    textcnn_model = Program.models.create_textcnn_model.textcnn_model(feature_dim)
    # print(textcnn_model.summary())
    # 特征缩放
    scaler = MinMaxScaler()  # 创建 MinMaxScaler 实例
    X = scaler.fit_transform(X)  # 进行训练集的缩放
    joblib.dump(scaler, 'scaler.pkl')  # 保存缩放器
    textcnn_aucs, textcnn_accs, textcnn_mccs, textcnn_f1s, textcnn_sns, textcnn_sps = ([] for _ in range(6))
    # all_train_accs, all_val_accs, all_train_losses, all_val_losses = ([] for _ in range(4))
    all_train_accs, all_val_accs, all_train_losses, all_val_losses = ([[] for _ in range(5)] for _ in
                                                                      range(4))  # 每折单独列表
    # 初始化四个列表，每个列表包含十个空列表用于十折交叉验证
    # all_train_accs = [[] for _ in range(10)]
    # all_val_accs = [[] for _ in range(10)]
    # all_train_losses = [[] for _ in range(10)]
    # all_val_losses = [[] for _ in range(10)]

    total_train_time = []
    total_train_time_0 = 0  # 用于存储总训练时间
    best_acc = 0
    best_fold = -1
    # KFold 交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    mem_before = p.memory_info().rss / (1024 * 1024)  # 计算初始内存使用
    for fold, (train_index, test_index) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        reshaped_X_train = np.expand_dims(X_train, axis=-1)
        reshaped_X_test = np.expand_dims(X_test, axis=-1)

        y_train_encoded = to_categorical(y_train, num_classes)
        y_test_encoded = to_categorical(y_test, num_classes)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # 记录 CPU 使用情况
        cpu_before = p.cpu_percent(interval=None)  # 初始化 CPU 使用率

        start_time = time.time()

        history = textcnn_model.fit(reshaped_X_train, y_train_encoded, epochs=num_epochs,
                                    batch_size=num_batch_size, verbose=0,
                                    validation_data=(reshaped_X_test, y_test_encoded),
                                    callbacks=[early_stopping])#

        end_time = time.time()
        total_train_time.append(end_time - start_time)
        total_train_time_0 += (end_time - start_time)  # 累加每折训练时间

        # # 记录训练和验证的准确率和损失

        all_train_accs[fold] = history.history['accuracy']
        all_val_accs[fold] = history.history['val_accuracy']
        all_train_losses[fold] = history.history['loss']
        all_val_losses[fold] = history.history['val_loss']

        predictions = textcnn_model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test_encoded, axis=1)

        acc = round(accuracy_score(true_classes, predicted_classes) * 100, 2)
        fpr, tpr, _ = roc_curve(true_classes, predictions[:, 1])
        auc_score = round(auc(fpr, tpr) * 100, 2)
        mcc = round(matthews_corrcoef(true_classes, predicted_classes) * 100, 2)
        f1 = round(f1_score(true_classes, predicted_classes, average='weighted') * 100, 2)
        cm = confusion_matrix(true_classes, predicted_classes)

        tn, fp, fn, tp = cm.ravel()
        sn = round(tp / (tp + fn) * 100, 2)
        sp = round(tn / (tn + fp) * 100, 2)

        textcnn_accs.append(acc)
        textcnn_aucs.append(auc_score)
        textcnn_mccs.append(mcc)
        textcnn_f1s.append(f1)
        textcnn_sns.append(sn)
        textcnn_sps.append(sp)
        # 记录结束时的 CPU 使用情况
        cpu_after = p.cpu_percent(interval=None)
        # 保存每折模型权重
        # textcnn_model.save_weights(f'fold_{fold+1}.weights.h5')
        if acc > best_acc:
            best_acc = acc
            best_fold = fold
        print(f'Fold {fold + 1} - Accuracy: {acc:.2f}, AUC: {auc_score:.2f}, MCC: {mcc:.2f}, F1 Score: {f1:.2f}, SN: {sn:.2f}, SP: {sp:.2f}, Epochs:{len(history.history['loss'])}')
        # 计算 CPU 使用率（在训练期间的平均值）
        avg_cpu_usage = (cpu_before + cpu_after) / 2
    # 使用所有数据训练模型并保存
    # reshaped_X = np.expand_dims(X, axis=-1)
    # y_encoded = to_categorical(y, num_classes)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # final_model = models.create_textcnn_model.textcnn_model(feature_dim)
    # final_model.fit(reshaped_X, y_encoded, epochs=num_epochs, batch_size=num_batch_size, verbose=1,callbacks=[early_stopping])
    #
    # # 保存最终模型
    # final_model.save('final_textcnn_model.keras')

    # 计算训练过程中的内存使用
    mem_after = p.memory_info().rss / (1024 * 1024)  # 计算结束时的内存使用
    memory_usage = mem_after - mem_before

    # 记录结束时的 CPU 使用情况
    cpu_after = p.cpu_percent(interval=None)

    avg_cpu_usage = (cpu_before + cpu_after) / 2
    metrics_df = pd.DataFrame({
        'ACC': textcnn_accs,
        'AUC': textcnn_aucs,
        'MCC': textcnn_mccs,
        'F1score': textcnn_f1s,
        'SN': textcnn_sns,
        'SP': textcnn_sps,
    })

    # 平均 5 折数据
    def average_metrics(metric_lists, max_epochs):
        padded_metrics = [
            metric + [metric[-1]] * (max_epochs - len(metric)) if len(metric) < max_epochs else metric[:max_epochs] for
            metric in metric_lists]
        return np.mean(padded_metrics, axis=0)

    avg_train_accs = average_metrics(all_train_accs, num_epochs)
    avg_val_accs = average_metrics(all_val_accs, num_epochs)
    avg_train_losses = average_metrics(all_train_losses, num_epochs)
    avg_val_losses = average_metrics(all_val_losses, num_epochs)
    # 计算平均指标
    mean_metrics = {
        'Accuracy': np.mean(textcnn_accs),
        'AUC': np.mean(textcnn_aucs),
        'MCC': np.mean(textcnn_mccs),
        'F1_SCORE': np.mean(textcnn_f1s),
        'SN': np.mean(textcnn_sns),
        'SP': np.mean(textcnn_sps),
        'Accuracy_std': np.std(textcnn_accs),
        'AUC_std': np.std(textcnn_aucs),
        'MCC_std': np.std(textcnn_mccs),
        'F1_SCORE_std': np.std(textcnn_f1s),
        'SN_std': np.std(textcnn_sns),
        'SP_std': np.std(textcnn_sps)
    }

    return {
        'train_accs': avg_train_accs,
        'val_accs': avg_val_accs,
        'train_losses': avg_train_losses,
        'val_losses': avg_val_losses,
        'total_train_time': total_train_time,
        'total_train_time_0': total_train_time_0,
        'mean_metrics': mean_metrics,
        'metrics_df': metrics_df,
        'avg_cpu_usage': avg_cpu_usage,
        'avg_memory_usage': memory_usage,
    }
# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

def plot_per_overall(all_results):
    for i, result in enumerate(all_results):
        plt.figure(figsize=(12, 6))

        # 绘制准确率曲线
        plt.subplot(1, 2, 1)
        plt.plot(result['train_accs'], label='Train Accuracy', color='blue')
        plt.plot(result['val_accs'], label='Validation Accuracy', color='orange')
        plt.title(f'Accuracy Curve (D1)',fontsize=14)
        plt.xlabel('Epochs',fontsize=14)
        plt.ylabel('Accuracy',fontsize=14)
        plt.legend(fontsize=14)

        # 绘制损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(result['train_losses'], label='Train Loss', color='blue')
        plt.plot(result['val_losses'], label='Validation Loss', color='orange')
        plt.title(f'Loss Curve (D2)',fontsize=14)
        plt.xlabel('Epochs',fontsize=14)
        plt.ylabel('Loss',fontsize=14)
        plt.legend(fontsize=14)

        plt.tight_layout()
        plt.savefig(f'Figures/overall_metrics_dataset_{i + 1}_SVM_LR_LDA_LIGHTGBM.png', dpi=300)  # 保存为不同的文件
        plt.close()  # 关闭当前图形，避免重叠
def plot_overall(all_results):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for i, result in enumerate(all_results):
        plt.plot(result['train_accs'], label=f'Train Accuracy (benchmark dataset)')
        plt.plot(result['val_accs'], label=f'Validation Accuracy (benchmark dataset)')
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.title('Overall Accuracy Curve (A)',fontsize=14)
    plt.xlabel('Epochs',fontsize=14)
    plt.ylabel('Accuracy',fontsize=14)
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, result in enumerate(all_results):
        plt.plot(result['train_losses'], label=f'Train Loss (benchmark dataset)')
        plt.plot(result['val_losses'], label=f'Validation Loss (benchmark dataset)')
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.title('Overall Loss Curve (B)',fontsize=14)
    plt.xlabel('Epochs',fontsize=14)
    plt.ylabel('Loss',fontsize=14)
    plt.legend()

    plt.tight_layout()
    plt.savefig('Figures/overall_metrics0528.png', dpi=300)
    plt.show()

# 绘制每个数据集的训练时间比对图
def plot_toal_training_time(all_results):
    total_train_times = [result['total_train_time_0'] for result in all_results]  # 确保提取为一维列表
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(total_train_times) + 1), total_train_times, color='skyblue')
    plt.title('Total Training Time for Each Dataset')
    plt.xlabel('Dataset Number')
    plt.ylabel('Total Time (seconds)')
    plt.xticks(range(1, len(total_train_times) + 1))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('total_training_time_comparison.png', dpi=300)
    plt.show()
def plot_training_time(all_results):
    for i, result in enumerate(all_results):
        plt.figure(figsize=(10, 5))
        plt.bar(range(1, len(result['total_train_time']) + 1), result['total_train_time'], color='skyblue')
        # plt.title(f'Training_dataset Time for Each Fold (Dataset {i + 1})')
        plt.title(f'Training Time for Each Fold (benchmark dataset)')
        plt.xlabel('Fold Number')
        plt.ylabel('Time (seconds)')
        plt.xticks(range(1, len(result['total_train_time']) + 1))
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'training_time_per_fold_dataset_{i + 1}.png', dpi=300)
        plt.show()
def plot_cor(data):
    # 计算相关性矩阵
    correlation_matrix = data.corr()
    # 排除目标列
    feature_columns = [col for col in data.columns if col != 'label']  # 替换 'label' 为您的目标列名
    # 选择与目标列相关性最大的前 20 个特征
    top_corr_features = correlation_matrix.loc[feature_columns, 'label'].nlargest(10).index

    # 绘制相关性热图
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(correlation_matrix.loc[top_corr_features, top_corr_features], annot=True, fmt=".2f", cmap='coolwarm', square=True)
    # 替换负号
    for text in ax.texts:
        if text.get_text().startswith('-'):
            text.set_text(text.get_text().replace('-', '− '))

    plt.savefig(f'Figures/Top 10 Correlation Heatmap.png',dpi=300)
    plt.xticks(rotation=45)  # 这里设置倾斜角度
    # plt.yticks(rotation=45)  # 这里设置倾斜角度
    plt.title('Top 10 Correlation Heatmap')
    plt.show()
# 数据集列表
datasets = [
    # '../Files/AllTrainingDataset_0405_NO_RF_GBDT_XGB_DNC.csv'
    # '../Files/AllTrainingDataset_0405_NO_RF_GBDT_XGB.csv'
    # '../Files/optimumAllTrainingDataset.csv'
    # '../Files/optimumAllTrainingDataset_F1_F2.csv'#,
    # '../Files/optimumAllTrainingDataset_NO_RF_GBDT_XGB.csv'#,
    # '../Files/optimumAllTrainingDataset_F1.csv',
    # '../Files/optimumAllTrainingDataset_F2.csv',
    # '../Files/optimumAllTrainingDataset_F3.csv',
    # '../Files/optimumAllTrainingDataset_F1_F2.csv',
    # '../Files/optimumAllTrainingDataset_F1_F3.csv',
    # '../Files/optimumAllTrainingDataset_F2_F3.csv',
    # '../Files/optimumAllTrainingDataset_F1_F2_F3.csv'
    # '../Files/AllTrainingDataset_0406.csv'
    # '../Files/AllTrainingDataset_0411_WITHOUT_LR.csv'
    # '../Files/AllTrainingDataset_0406_NO_RF_GBDT_XGB.csv',
    # '../Files/AllTrainingDataset_0411_WITHOUT.csv'
    # '../Files/AllTrainingDataset_0411_WITH_SVM.csv',
    # '../Files/AllTrainingDataset_0411_WITH_SVM_LR.csv',
    # '../Files/AllTrainingDataset_0411_WITH_SVM_LR_LDA.csv',
    # '../Files/AllTrainingDataset_0411_WITH_ALL.csv'
    # '../Files/AllTrainingDataset_0413_WITHOUT.csv',
    # '../Files/AllTrainingDataset_0413_WITH_SVM.csv',
    # '../Files/AllTrainingDataset_0413_WITH_SVM_LR.csv',
    # '../Files/AllTrainingDataset_0413_WITH_SVM_LR_LDA.csv',
    # '../Files/AllTrainingDataset_0413_WITH_ALL.csv'
    # '../Files/final/AllTrainingDataset_0518_WITH_ALL.csv'
    # '../Files/final/AllTrainingDataset_0518_WITH_F1.csv'
    # '../Files/final/AllTrainingDataset_0518_WITH_SVM.csv'
    # '../Files/final/AllTrainingDataset_0518_WITH_SVM_LR.csv'
    # '../Files/final/AllTrainingDataset_0518_WITH_SVM_LR_LDA.csv'
    '../Files/final/AllTrainingDataset_0518_WITH_SVM_LR_LDA_LIGHTGBM.csv'
]

# 初始化结果存储
all_results = []

# 遍历每个数据集并训练模型
for dataset in datasets:
    results = training_model(dataset)
    all_results.append(results)
    # 打印每个数据集训练完后的总训练时间
    print(f'Dataset: {dataset}, Total Training Time: {results["total_train_time_0"]:.2f} seconds')
# 在所有数据集训练完成后绘图
for result in all_results:
    mean_metrics = result['mean_metrics']
    # 按行打印每个评价指标均值
    print("评价指标均值及标准差:")
    for metric, value in mean_metrics.items():
        print(f"{metric}: {value:.2f}")  # 保留两位小数
plot_overall(all_results)
# 记录结束时的 CPU 和内存使用情况
# cpu_after = p.cpu_percent(interval=None)
# mem_after = p.memory_info().rss / (1024 * 1024)
# memory_usage = mem_after - mem_before
#
# print(f'Memory Usage: {memory_usage:.2f} MB')
# print(f'Average CPU Usage: {results["avg_cpu_usage"]:.2f}%')
# print(f'Average CPU Usage: {results["avg_memory_usage"]:.2f}MB')
plot_per_overall(all_results)
