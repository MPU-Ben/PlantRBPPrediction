import time

from process import read,getFeatures,getAdditionFeatures

start_time=time.time()
print('Start Time: ',start_time)

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
S = getFeatures.gF("PROT",X_,Y_,**args)

X = S[:, :-1]
y = S[:, -1]

T = getAdditionFeatures.getAdditionFeatures(X,y)

T.to_csv("files/fullIndependenceTestingDataset.csv",index=False)
T.to_csv("files/optimumAllIndependenceTestingDataset.csv",index=False)
# AllDataPath='files/optimumAllIndependenceTestingDataset.csv'

# print('-------train_test_split---------')
# X_train,X_test,y_train,y_test=train_test_split(X,y,shuffle=True,test_size=0.3,stratify=y)#,random_state=40

end_time=time.time()
print('End Time: ',end_time)
print('Time-Consume: ',(end_time-start_time)/60)
