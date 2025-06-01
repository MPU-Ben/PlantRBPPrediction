import lightgbm
import numpy as np
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

    # svm
    model = svm.SVC(probability=True,kernel='rbf', gamma='auto', C=10)
    model.fit(X, y)
    # 获取预测概率
    y_pred_svm_prob = model.predict_proba(X)[:, 1]
    y_pred_prob_svm_df = pd.DataFrame(y_pred_svm_prob, columns=['svm_pred'])

    # LGBMClassifier
    model = lightgbm.LGBMClassifier(num_leaves=63, n_estimators=200, max_depth= 5, learning_rate=0.2)
    model.fit(X, y)
    y_pred_prob_light = model.predict_proba(X)[:, 1]
    y_pred_prob_light_df = pd.DataFrame(y_pred_prob_light, columns=['lightgbm_pred'])

    # LogisticRegression
    model = LogisticRegression(solver='liblinear', C= 100)
    model.fit(X, y)
    y_pred_prob_lr = model.predict_proba(X)[:, 1]
    y_pred_prob_lr_df = pd.DataFrame(y_pred_prob_lr, columns=['lr_pred'])

    # LDA
    model = LinearDiscriminantAnalysis(solver='svd', priors=None)
    model.fit(X, y)
    y_pred_prob_lda = model.predict_proba(X)[:, 1]
    y_pred_prob_lda_df = pd.DataFrame(y_pred_prob_lda, columns=['lr_pred'])

    X_Columns = np.array(X_Columns).flatten()
    y_Column = np.array(y_Column).flatten()
    X_df = pd.DataFrame(X, columns = X_Columns)
    y_df = pd.DataFrame(y, columns = y_Column)
    T = pd.concat([X_df,y_pred_prob_svm_df, y_pred_prob_light_df,y_pred_prob_lr_df,y_pred_prob_lda_df,y_df], axis=1)
    return T