import time
from process import runClassifiers


start_time=time.time()
print('开始时间：',start_time)

# 使用经过特征选择的全部数据集
optimumAllDataPath='files/optimumAllTrainingDataset_F0_F1_F2.csv'

# 在特征选择后的数据集上训练多个分类器
print('-----------训练多个分类器-----------')
argsForClassifier={'nFCV':5,'dataset':optimumAllDataPath,'auROC':1,'boxPlot':1,'accPlot':1}

runClassifiers.runClassifiers(argsForClassifier)

end_time=time.time()
print('结束时间：',end_time)
print('总耗时：',(end_time-start_time) / 60)