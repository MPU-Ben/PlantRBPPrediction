import time

import lightgbm
import pandas as pd
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split

from process import save, read,selectedImportantFeatures,getFeatures,getAdditionFeatures

start_time=time.time()
print('开始时间：',start_time)

file ="data/Training/training_dataset/ALL_RBP_nonRBP_data_4992.fa"
label ="data/Training/training_dataset/ALL_RBP_nonRBP_labels_4992.txt"

# file ="data/Training/testing_dataset/ALL_RBP_NONRBP_1086_TestingData.fa"
# label ="data/Training/testing_dataset/ALL_RBP_NONRBP_1086_TestingData_labels.txt"

pos_neg_seq= read.fetchX(file, "PROT")

args={'onehot':0,'length':0,'zCurve':0,'gcContent':0,'cumulativeSkew':0,'atgcRatio':0,'pseudoKNC':0,
      'monoMono':0,'monoDi':0,'diMono':0,'monoTri':0,'diDi':0,'diTri':0,'triMono':0,'triDi':0,
      'PseEIIP':0,'EIIP':0,'NAC':0,'DNC':0,'TNC':0,'AAC':1,'DPC':1,
      'CT':1,'DRA':0,'DAC':0,'kmer_freq_o':0,'kmer_freq_t':0,'MaxORFsLen':0,'orf_coverage':0,'count_orfs':0,
       'kGap':0,'kTupe':0}
X_,Y_= read.fetchXY(file, label)
print('-------开始提取特征-----------')
S = getFeatures.gF("PROT",X_,Y_,**args)
# 划分特征选择前的数据集
X = S[:, :-1]
y = S[:, -1]

T = getAdditionFeatures.getAdditionFeatures(X,y)
# X = T[:, :-1]
# y = T[:, -1]

T.to_csv("files/fullIndependenceTestingDataset.csv",index=False)
# save.saveCSV(X, y, 'full')

# X=selectedImportantFeatures.importantFeaturesByDT(X,y)
# 保存特征选择后的全部数据集
# save.saveCSV(X,y,'optimumAllDataset')
T.to_csv("files/optimumAllIndependenceTestingDataset.csv",index=False)
# AllDataPath='files/optimumAllIndependenceTestingDataset.csv'
#
# X = T.drop('label', axis=1)  # 特征
# y = T['label']               # 目标变量
#
# print('-------划分训练集和测试集---------')
# X_train,X_test,y_train,y_test=train_test_split(X,y,shuffle=True,test_size=0.3,stratify=y)#,random_state=40
#
# print("X_train.shape",X_train.shape)
# print("Y_train.shape",y_train.shape)

# 保存未经特征选择测试集
# save.saveCSV(X_test,y_test,'models')

# 保存经过特征选择的训练集
# train_df = pd.concat([X_train, y_train], axis=1)
# optimumTrainDataPath='files/optimumTrainDataset.csv'
# train_df.to_csv(optimumTrainDataPath,index=False)
# # 保存经过特征选择的测试集
# test_df = pd.concat([X_test, y_test], axis=1)
# optimumTestDataPath='files/optimumTestDataset.csv'
# test_df.to_csv(optimumTestDataPath,index=False)
end_time=time.time()
print('结束时间：',end_time)
print('总耗时：',(end_time-start_time)/60)