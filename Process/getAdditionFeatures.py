import lightgbm
import numpy as np
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier,GradientBoostingClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
def getAdditionFeatures(X_, y_):
    import pandas as pd
    from xgboost import XGBClassifier

    X_df = pd.DataFrame(X_)
    y_df = pd.DataFrame(y_)
    X_Columns = X_df.iloc[:1,:].values
    y_Column = y_df.iloc[:1,:].values
    X = X_df.iloc[1:,:].values
    y = y_df.iloc[1:,-1].values

    # 将字符串标签转换为整数标签
    le = LabelEncoder()
    y = le.fit_transform(y)
    # 训练XGBoost模型
    model = XGBClassifier()
    model.fit(X, y)
    # 获取预测概率
    y_pred_xgb_prob = model.predict_proba(X)[:, 1]
    y_pred_prob_xgb_df = pd.DataFrame(y_pred_xgb_prob, columns=['xgb_pred'])

    # svm
    model = svm.SVC(probability=True)
    model.fit(X, y)
    # 获取预测概率
    y_pred_svm_prob = model.predict_proba(X)[:, 1]
    y_pred_prob_svm_df = pd.DataFrame(y_pred_svm_prob, columns=['svm_pred'])

    #RandomForest
    model = RandomForestClassifier()
    model.fit(X, y)
    y_pred_prob_rf = model.predict_proba(X)[:, 1]
    y_pred_prob_rf_df = pd.DataFrame(y_pred_prob_rf, columns=['rf_pred'])

    #GBDT
    model = GradientBoostingClassifier()
    model.fit(X, y)
    y_pred_prob_gbdt = model.predict_proba(X)[:, 1]
    y_pred_prob_gbdt_df = pd.DataFrame(y_pred_prob_gbdt, columns=['gbdt_pred'])
    # BaggingClassifier
    # model = BaggingClassifier()
    # model.fit(X, y)
    # y_pred_prob_bg = model.predict_proba(X)[:, 1]
    # y_pred_prob_bg_df = pd.DataFrame(y_pred_prob_bg, columns=['bg_pred'])

    # LGBMClassifier
    model = lightgbm.LGBMClassifier()
    model.fit(X, y)
    y_pred_prob_light = model.predict_proba(X)[:, 1]
    y_pred_prob_light_df = pd.DataFrame(y_pred_prob_light, columns=['lightgbm_pred'])

    # LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)
    y_pred_prob_lr = model.predict_proba(X)[:, 1]
    y_pred_prob_lr_df = pd.DataFrame(y_pred_prob_lr, columns=['lr_pred'])

    # LDA
    model = LinearDiscriminantAnalysis()
    model.fit(X, y)
    y_pred_prob_lda = model.predict_proba(X)[:, 1]
    y_pred_prob_lda_df = pd.DataFrame(y_pred_prob_lda, columns=['lr_pred'])

    X_Columns = np.array(X_Columns).flatten()
    y_Column = np.array(y_Column).flatten()
    X_df = pd.DataFrame(X, columns = X_Columns)
    y_df = pd.DataFrame(y, columns = y_Column)
    T = pd.concat([X_df,y_pred_prob_svm_df, y_pred_prob_rf_df,y_pred_prob_gbdt_df,y_pred_prob_light_df,y_pred_prob_xgb_df,y_pred_prob_lr_df,y_pred_prob_lda_df,y_df], axis=1)
    return T
# import time
# import numpy as np
# import pandas as pd
# import psutil
# from sklearn import svm
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder
#
#
# def getAdditionFeatures(X_, y_):
#     X_df = pd.DataFrame(X_)
#     y_df = pd.DataFrame(y_)
#     X_Columns = X_df.iloc[:1, :].values
#     y_Column = y_df.iloc[:1, :].values
#     X = X_df.iloc[1:, :].values
#     y = y_df.iloc[1:, -1].values
#
#     # 将字符串标签转换为整数标签
#     le = LabelEncoder()
#     y = le.fit_transform(y)
#
#     # 获取当前进程的 PID
#     pid = psutil.Process().pid
#     p = psutil.Process(pid)
#
#     # 记录模型训练时间和资源使用情况
#     results = {}
#
#     # 定义一个函数来训练模型并监控内存和 CPU 使用
#     def train_model(model_class, model_name):
#         before_memory = p.memory_info().rss / (1024 * 1024)
#
#         start_time = time.time()
#         model = model_class()
#         model.fit(X, y)
#         end_time = time.time()
#
#         after_memory = p.memory_info().rss / (1024 * 1024)
#         memory_usage = after_memory - before_memory
#         training_time = end_time - start_time
#
#         # 直接用 interval=1 阻塞采集 CPU 使用率，并根据核数校正
#         cpu_usage_raw = p.cpu_percent(interval=1)
#         cpu_usage = cpu_usage_raw / psutil.cpu_count(logical=True)
#
#         return model, training_time, memory_usage, cpu_usage
#
#     # SVM
#     model, training_time, memory_usage, cpu_usage = train_model(lambda: svm.SVC(probability=True), 'SVM')
#     results['SVM'] = {
#         'time': training_time,
#         'mem_usage': memory_usage,
#         'cpu_usage': cpu_usage
#     }
#     y_pred_svm_prob = model.predict_proba(X)[:, 1]
#     y_pred_prob_svm_df = pd.DataFrame(y_pred_svm_prob, columns=['svm_pred'])
#
#     # Random Forest
#     model, training_time, memory_usage, cpu_usage = train_model(RandomForestClassifier, 'RandomForest')
#     results['RandomForest'] = {
#         'time': training_time,
#         'mem_usage': memory_usage,
#         'cpu_usage': cpu_usage
#     }
#     y_pred_prob_rf = model.predict_proba(X)[:, 1]
#     y_pred_prob_rf_df = pd.DataFrame(y_pred_prob_rf, columns=['rf_pred'])
#
#     # Logistic Regression
#     model, training_time, memory_usage, cpu_usage = train_model(LogisticRegression, 'LogisticRegression')
#     results['LogisticRegression'] = {
#         'time': training_time,
#         'mem_usage': memory_usage,
#         'cpu_usage': cpu_usage
#     }
#     y_pred_prob_lr = model.predict_proba(X)[:, 1]
#     y_pred_prob_lr_df = pd.DataFrame(y_pred_prob_lr, columns=['lr_pred'])
#
#     # LDA
#     model, training_time, memory_usage, cpu_usage = train_model(LinearDiscriminantAnalysis, 'LDA')
#     results['LDA'] = {
#         'time': training_time,
#         'mem_usage': memory_usage,
#         'cpu_usage': cpu_usage
#     }
#     y_pred_prob_lda = model.predict_proba(X)[:, 1]
#     y_pred_prob_lda_df = pd.DataFrame(y_pred_prob_lda, columns=['lda_pred'])
#
#     # 整合结果
#     X_Columns = np.array(X_Columns).flatten()
#     y_Column = np.array(y_Column).flatten()
#     X_df = pd.DataFrame(X, columns=X_Columns)
#     y_df = pd.DataFrame(y, columns=y_Column)
#     T = pd.concat([X_df,
#                    y_pred_prob_svm_df,
#                    y_pred_prob_rf_df,
#                    y_pred_prob_lr_df,
#                    y_pred_prob_lda_df,
#                    y_df], axis=1)
#
#     # 打印每个模型的训练时间、内存及 CPU 使用情况
#     for model_name, metrics in results.items():
#         print(f"{model_name}:")
#         print(f"  Training Time: {metrics['time']:.4f} seconds")
#         print(f"  Memory Usage: {metrics['mem_usage']:.2f} MB")
#         print(f"  CPU Usage: {metrics['cpu_usage']:.2f}%")  # 输出 CPU 占用率
#
#     return T
