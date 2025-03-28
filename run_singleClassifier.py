import time
from process import trainModel,evaluateModel

start_time=time.time()
print('开始时间：',start_time)

# 保存经过特征选择的训练集
optimumTrainDataPath='files/optimumAllTrainDataset.csv'
# 保存经过特征选择的测试集
optimumTestDataPath='files/optimumAllTestDataset.csv'

# 在特征选择后的训练集上训练单个模型
print('------------训练单个分类器-----------')
choices={'LR':0,'SVM':0,'KNN':0,'DT':0,'NB':0,'Bagging':0,'RF':0,'AB':0,'GB':0,'LDA':0,'ET':0,'XGB':1,'VOTE':0,'LGBM':0,'LSTM':0}
argsForTrain={'optimunTrainDataPath':optimumTrainDataPath,'optimunTestDataPath':optimumTestDataPath,'model':choices}
trainModel.train(argsForTrain)

# 使用特征选择后的训练集和测试及评估模型
print('---------评估分类器-----------')
argsForEval={'optimumTrainDatasetPath':optimumTrainDataPath,'optimumtestDatasetPath':optimumTestDataPath,'splited':1}
evaluateModel.evaluate(argsForEval)

end_time=time.time()
print('结束时间：',end_time)
print('总耗时：',(end_time-start_time) / 60)