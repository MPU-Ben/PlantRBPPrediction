# import itertools
import numpy as np
from collections import Counter
# from Bio.Seq import Seq
from Bio.Data.CodonTable import unambiguous_rna_by_name
# import iFeatureOmegaCLI
from Bio.Seq import Seq
# from pyrna.feature_extraction import PseKNCExtractor
# from Bio.SeqUtils.ProtParamData import kd

DNAelements = 'ACGT'
RNAelements = 'ACGU'
proteinElements = 'ACDEFGHIKLMNPQRSTVWY'
trackingFeatures = []
T = []
t=[]
def sequenceType(seqType):
    if seqType == 'DNA':
        elements = DNAelements
    else:
        if seqType == 'RNA':
            elements = RNAelements
        else:
            if seqType == 'PROTEIN' or seqType == 'PROT':
                elements = proteinElements
            else:
                elements = None

    return elements
def gF(seqType, X, Y, **args):
    T.clear()

    def kmer_freq_o(seq,k):
        # 划分RNA序列为k-mer
        k = 1
        kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]

        # 计算k-mer频率
        kmer_counts = dict(Counter(kmers))
        sequence_length = len(seq)
        kmer_frequencies = [count / sequence_length for count in kmer_counts.values()]

        # 计算最接近的4的k次方的长度
        max_length = int(np.ceil(len(kmer_frequencies) / 4 ** k) * 4 ** k)

        # 填充特征向量
        padded_feature_vector = np.zeros(max_length)
        padded_feature_vector[:len(kmer_frequencies)] = kmer_frequencies
        # return padded_feature_vector
        t.extend(list(padded_feature_vector))
    def kmer_freq_t(seq,k):
        # 划分RNA序列为k-mer
        k = 2
        kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]

        # 计算k-mer频率
        kmer_counts = dict(Counter(kmers))
        sequence_length = len(seq)
        kmer_frequencies = [count / sequence_length for count in kmer_counts.values()]

        # 计算最接近的4的k次方的长度
        max_length = int(np.ceil(len(kmer_frequencies) / 4 ** k) * 4 ** k)

        # 填充特征向量
        padded_feature_vector = np.zeros(max_length)
        padded_feature_vector[:len(kmer_frequencies)] = kmer_frequencies
        # return padded_feature_vector
        t.extend(list(padded_feature_vector))

    def zCurve(x, seqType):
        tmp_t = []
        tmp_x = []
        tmp_y = []
        tmp_z = []
        feature_names = ['x_axis','y_axis','z_axis']
        tmp_t.append(feature_names)
        for sequence in x:
            if seqType == 'DNA' or seqType == 'RNA':
                if seqType == 'DNA':
                    TU = sequence.count('T')
                else:
                    if seqType == 'RNA':
                        TU = sequence.count('U')
                    else:
                        None

                A = sequence.count('A'); C = sequence.count('C'); G = sequence.count('G');

                x_ = (A + G) - (C + TU)
                y_ = (A + C) - (G + TU)
                z_ = (A + TU) - (C + G)
                # t.append(x_); t.append(y_); t.append(z_)
                tmp_x.append(x_)
                tmp_y.append(y_)
                tmp_z.append(z_)

        tmp = [[x , y , z] for x, y, z in zip(tmp_x,tmp_y,tmp_z)]
        tmp_t.extend(tmp)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
        # print()
    def gcContent(x, seqType):
        tmp_t = []
        feature_names = ['gcContent']
        tmp_t.append(feature_names)
        for sequence in x:
            if seqType == 'DNA' or seqType == 'RNA':
                if seqType == 'DNA':
                    TU = sequence.count('T')
                else:
                    if seqType == 'RNA':
                        TU = sequence.count('U')
                    else:
                        None

            A = sequence.count('A')
            C = sequence.count('C')
            G = sequence.count('G')
            feature_vector = [(G + C) / (A + C + G + TU)  * 100.0]
            tmp_t.append(feature_vector)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
        # print()
                # trackingFeatures.append('gcCotent')
    def cumulativeSkew(x, seqType):

        if seqType == 'DNA' or seqType == 'RNA':

            if seqType == 'DNA':
                TU = x.count('T')
            else:
                if seqType == 'RNA':
                    TU = x.count('U')
                else:
                    None

            A = x.count('A');
            C = x.count('C');
            G = x.count('G');

            GCSkew = (G-C)/(G+C)
            ATSkew = (A-TU)/(A+TU)

            t.append(GCSkew)
            t.append(ATSkew)

            trackingFeatures.append('GCSkew')
            trackingFeatures.append('ATSkew')
    def atgcRatio(x, seqType):
        tmp_t = []
        feature_names = ['atgcRatio']
        tmp_t.append(feature_names)
        for sequence in x:
            if seqType == 'DNA' or seqType == 'RNA':

                if seqType == 'DNA':
                    TU = sequence.count('T')
                else:
                    if seqType == 'RNA':
                        TU = sequence.count('U')
                    else:
                        None

            A = sequence.count('A')
            C = sequence.count('C')
            G = sequence.count('G')
            feature_vector = [(A + TU) / (G + C)]
            tmp_t.append(feature_vector)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
        # print()
                # trackingFeatures.append('atgcRatio')
    def NAC(x):
        NA = 'ACGU'
        tmp_t = []  # 存储特征矩阵
        feature_names = [na for na in NA]#['NAC-' + na for na in NA]
        tmp_t.append(feature_names)
        # t.append(feature_names)
        for sequence in x:
            sequence = sequence.strip() #x.strip()
            count = Counter(sequence)
            feature_vector = []

            for na in NA:
                count_na = count[na] / len(sequence)
                feature_vector.append(count_na)
            tmp_t.append(feature_vector)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T)>0 else tmp_t
        # return tmp_t
        # print()
    def DNC(x):

        AA = 'ACGU'
        AADict = {}
        code = [0] * 16
        feature_names = []
        feature_values = []
        tmp_t = []
        # 生成特征名
        for i in range(len(code)):
            feature_name = AA[i // 4] + AA[i % 4]#.join(AA[i // 4] + AA[i % 4])'DNC-' + ''.join()
            feature_names.append(feature_name)
        # t.append(feature_names)
        tmp_t.append(feature_names)

        for i in range(len(AA)):
            AADict[AA[i]] = i

        for sequence in x:
            sequence = sequence.strip()
            tmpCode = [0] * 16
            feature_vector = []
            try:
                for j in range(len(sequence) - 2 + 1):
                    tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j + 1]]] += 1
                    # print('{0}-{1}'.format(sequence[j] + sequence[j + 1], str(AADict[sequence[j]] * 4 + AADict[sequence[j + 1]])))

                if sum(tmpCode) != 0:
                    for i in tmpCode:
                        calcs = i/sum(tmpCode)
                        feature_vector.append(calcs)
                    feature_values.append(feature_vector)

                    tmp_t.append(feature_vector)
            except:
                print(sequence)
        # tmp_t = np.array(tmp_t)
        # t.append(tmp_t,)
        global T
        # t = [t[i] + row for i, row in enumerate(tmp_t)]
        # 按列合并 tmp_t 到 t
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
        # np.concatenate((T,tmp_t),axis=1)
        # return tmp_t
        # print()
    # 三核苷酸组成(Trinucleotide composition, TNC)
    def TNC(x):
        AA = 'ACGU'
        AADict = {}
        Code = [0] * 64
        feature_names = []
        feature_values = []
        tmp_t = []

        # 生成特征名
        for i in range(len(Code)):
            feature_name = ''.join([AA[(i >> (2 * j)) & 3] for j in range(2, -1, -1)])
            feature_names.append(feature_name)
        tmp_t.append(feature_names)

        for i in range(len(AA)):
            AADict[AA[i]] = i

        for sequence in x:
            sequence = sequence.strip()
            tmpCode = [0] * 64
            feature_vector = []
            try:
                for j in range(len(sequence) - 3 + 1):
                    tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j + 1]] * 4 + AADict[sequence[j + 2]]] += 1
                if sum(tmpCode) != 0:
                    for i in tmpCode:
                        calcs = i / sum(tmpCode)
                        feature_vector.append(calcs)
                    feature_values.append(feature_vector)
                    # t.append(feature_vector)
                    tmp_t.append(feature_vector)
                    # trackingFeatures.append(feature_names)
            except:
                print(sequence)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t

        # print()
    def getSeqLength(x):
        tmp_t = []
        feature_name = ['SeqLength']
        tmp_t.append(feature_name)
        for sequence in x:
            seq = Seq(sequence)
            feature_vector = [len(seq)]
            tmp_t.append(feature_vector)

        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
    def PseKNC(x):
        k = 2 #间隔
        lambda_value = 3
        w = 0.5 #权重
        tmp_t = []
        feature_names = []
        AA = 'ACGU'
        AADict = {}
        code = [0] * 16
        for i in range(len(code)):
            feature_name = AA[i // 4] + AA[i % 4]
            feature_names.append('PseKNC_' + feature_name)
        for i in range(lambda_value):
            feature_name = 'lambda_' + str(i+1)
            feature_names.append(feature_name)

        tmp_t.append(feature_names)

        RNA_DINUCLEOTIDE_PARAMS = {
            'AA': {'Shift': -0.08, 'Slide': -1.27, 'Rise': 3.18, 'Twist': 31.0, 'Tilt': -0.8, 'Roll': 7.0},
            'UU': {'Shift': -0.08, 'Slide': -1.27, 'Rise': 3.18, 'Twist': 31.0, 'Tilt': -0.8, 'Roll': 7.0},
            'AC': {'Shift': 0.23, 'Slide': -1.43, 'Rise': 3.24, 'Twist': 32.0, 'Tilt': 0.8, 'Roll': 4.8},
            'GU': {'Shift': 0.23, 'Slide': -1.43, 'Rise': 3.24, 'Twist': 32.0, 'Tilt': 0.8, 'Roll': 4.8},
            'AG': {'Shift': -0.04, 'Slide': -1.50, 'Rise': 3.30, 'Twist': 30.0, 'Tilt': 0.5, 'Roll': 8.5},
            'CU': {'Shift': -0.04, 'Slide': -1.50, 'Rise': 3.30, 'Twist': 30.0, 'Tilt': 0.5, 'Roll': 8.5},
            'AU': {'Shift': -0.06, 'Slide': -1.36, 'Rise': 3.24, 'Twist': 33.0, 'Tilt': 1.1, 'Roll': 7.1},
            'CA': {'Shift': 0.11, 'Slide': -1.46, 'Rise': 3.09, 'Twist': 31.0, 'Tilt': 1.0, 'Roll': 9.9},
            'UG': {'Shift': 0.11, 'Slide': -1.46, 'Rise': 3.09, 'Twist': 31.0, 'Tilt': 1.0, 'Roll': 9.9},
            'CC': {'Shift': -0.01, 'Slide': -1.78, 'Rise': 3.32, 'Twist': 32.0, 'Tilt': 0.3, 'Roll': 8.7},
            'CG': {'Shift': 0.3,   'Slide': -1.89, 'Rise': 3.3, 'Twist': 27.0, 'Tilt': -0.1,  'Roll': 12.1},
            'GA': {'Shift': 0.07, 'Slide': -1.7, 'Rise': 3.38, 'Twist': 32.0, 'Tilt': 1.3, 'Roll': 9.4},
            'UC': {'Shift': 0.07, 'Slide': -1.7, 'Rise': 3.38, 'Twist': 32.0, 'Tilt': 1.3, 'Roll': 9.4},
            'GC': {'Shift': 0.07, 'Slide': -1.39, 'Rise': 3.22, 'Twist': 35.0, 'Tilt': 0.0, 'Roll': 6.1},
            'GG': {'Shift': -0.01, 'Slide': -1.78, 'Rise': 3.32, 'Twist': 32.0, 'Tilt': 0.3, 'Roll': 12.1},
            'UA': {'Shift': -0.02, 'Slide': -1.45, 'Rise': 3.26, 'Twist': 32.0,  'Tilt': -0.2, 'Roll': 10.7}
        }
        for sequence in x:
            # 获取 RNA 序列长度
            L = len(sequence)

            # 计算 RNA 序列的 k-mer 出现频率
            kmer_count = {}
            for i in range(L - k + 1):
                kmer = sequence[i:i + k]
                if kmer in kmer_count:
                    kmer_count[kmer] += 1
                else:
                    kmer_count[kmer] = 1
            # for kmer in sorted(kmer_count):
            #     feature_names.append(kmer)
            # ks = 0
            # 计算 k-mer 出现频率
            # kmer_freq = {ks: v / (L - int(k) + 1) for ks, v in kmer_count.items()}
            kmer_freq = {ks: (v / (L - int(k) + 1)) * (1 - w * sum([
                abs(RNA_DINUCLEOTIDE_PARAMS.get(ks[:2], {param: 0 for param in
                                                         ['Shift', 'Slide', 'Rise', 'Twist', 'Tilt', 'Roll']})[param] -
                    RNA_DINUCLEOTIDE_PARAMS.get(ks[1:], {param: 0 for param in
                                                         ['Shift', 'Slide', 'Rise', 'Twist', 'Tilt', 'Roll']})[param])
                for param in ['Shift', 'Slide', 'Rise', 'Twist', 'Tilt', 'Roll']
            ])) for ks, v in kmer_count.items()}

            # 计算二核苷酸的位置相关因子
            theta = []
            for j in range(1, lambda_value + 1):
                theta_sum = 0
                for i in range(L - j):
                    dinucleotide1 = sequence[i:i + 2]
                    dinucleotide2 = sequence[i + j:i + j + 2]
                    if dinucleotide1 in RNA_DINUCLEOTIDE_PARAMS and dinucleotide2 in RNA_DINUCLEOTIDE_PARAMS:
                        theta_sum += abs(sum(
                            [RNA_DINUCLEOTIDE_PARAMS[dinucleotide1][param] - RNA_DINUCLEOTIDE_PARAMS[dinucleotide2][param]
                             for param in ['Shift', 'Slide', 'Rise', 'Twist', 'Tilt', 'Roll']]))
                theta.append(theta_sum / (L - j))

            # 计算 PseDNC 特征向量
            feature_vector = [kmer_freq.get(kmer,0) for kmer in feature_names[:4 ** k]]
            feature_vector.extend([w * theta_j for theta_j in theta])
            # feature_vector.insert(0, feature_names)
            tmp_t.append(feature_vector)

        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
    def MaxORFsLen(x):
        tmp_t = []
        feature_names = ['MaxORFsLen']
        tmp_t.append([feature_names])
        # 使用通用 RNA 遗传密码表
        codon_table = unambiguous_rna_by_name["Standard"]

        for sequence in x:
            seq = Seq(sequence)
        # 定义起始和终止密码子
        #     start_codon = "ATG"
        #     stop_codons = ["TAA", "TAG", "TGA"]
            # 提取最长ORF的长度作为数值特征
            max_orf_length = 0
            current_length = 0
            in_orf = False

            for i in range(len(seq) - 2):
                codon = str(seq[i:i + 3])

                if not in_orf and codon in codon_table.start_codons:
                    in_orf = True
                    current_length = 3
                elif in_orf:
                    if codon in codon_table.stop_codons:
                        in_orf = False
                        if current_length > max_orf_length:
                            max_orf_length = current_length
                        current_length = 0
                    else:
                        current_length += 3
            # 检查最后一个 ORF
            if in_orf and current_length > max_orf_length:
                max_orf_length = current_length
            feature_vector = [max_orf_length]
            tmp_t.append([feature_vector])

        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
    def orf_coverage(x):
        tmp_t = []
        feature_names = ['ORFs_Coverage']
        tmp_t.append(feature_names)
        codon_table = unambiguous_rna_by_name["Standard"]
        start_codons = codon_table.start_codons  # ["AUG"]
        stop_codons = codon_table.stop_codons  # ["UAA", "UAG", "UGA"]

        for sequence in x:
            # 转换为RNA序列
            # rna_sequence = sequence.replace('T', 'U')
            seq = Seq(sequence)
            # 计算正向链上的ORF覆盖率
            forward_coverage = 0
            for i in range(len(seq) - 2):
                codon = str(seq[i:i + 3])
                if codon in start_codons:
                    j = i + 3
                    while j < len(seq) - 2:
                        codon = str(seq[j:j + 3])
                        if codon in stop_codons:
                            forward_coverage += j - i + 3
                            break
                        j += 3

            # # 计算反向链上的ORF覆盖率
            # reverse_seq = seq.reverse_complement_rna()
            # reverse_coverage = 0
            # for i in range(len(reverse_seq) - 2):
            #     codon = str(reverse_seq[i:i + 3])
            #     if codon in start_codons:
            #         j = i + 3
            #         while j < len(reverse_seq) - 2:
            #             codon = str(reverse_seq[j:j + 3])
            #             if codon in stop_codons:
            #                 reverse_coverage += j - i + 3
            #                 break
            #             j += 3

            # 计算双链的ORF覆盖率
            orf_coverage = forward_coverage #+ reverse_coverage
            feature_vector = [orf_coverage  / (len(seq))]
            tmp_t.append(feature_vector)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
        # print()
    def count_orfs(x):
        tmp_t = []
        feature_names = ['ORFs_Count']
        tmp_t.append(feature_names)
        codon_table = unambiguous_rna_by_name["Standard"]
        start_codons = codon_table.start_codons  # ["AUG"]
        stop_codons = codon_table.stop_codons  # ["UAA", "UAG", "UGA"]

        for sequence in x:
            orf_count = 0
            seq = Seq(sequence)
            # 搜索正向链
            for i in range(len(seq) - 2):
                codon = str(seq[i:i + 3])
                if codon in start_codons:
                    j = i + 3
                    while j < len(seq) - 2:
                        codon = str(seq[j:j + 3])
                        if codon in stop_codons:
                            orf_count += 1
                            break
                        j += 3

            # # 搜索反向链
            # reverse_seq = seq.reverse_complement_rna()
            # for i in range(len(reverse_seq) - 2):
            #     codon = str(reverse_seq[i:i + 3])
            #     if codon in start_codons:
            #         j = i + 3
            #         while j < len(reverse_seq) - 2:
            #             codon = str(reverse_seq[j:j + 3])
            #             if codon in stop_codons:
            #                 orf_count += 1
            #                 break
            #             j += 3

            # feature_vector = [orf_count / (len(x) * 2)]
            # feature_vector = [orf_count / len(seq)]
            feature_vector = [orf_count]
            tmp_t.append(feature_vector)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
        # print()
    #def getMFE():

    def AAC(x):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        tmp_t = []  # 存储特征矩阵
        feature_names = [na for na in AA]#['NAC-' + na for na in NA]
        tmp_t.append(feature_names)
        for sequence in x:
            sequence = sequence.strip()
            count = Counter(sequence)
            feature_vector = []

            for aa in AA:
                count_aa = count[aa] / len(sequence)
                feature_vector.append(count_aa)
            tmp_t.append(feature_vector)

        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t

    # 二肽组成(Di-peptide Compsition, DPC)
    def DPC(X):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        code = [0] * 400
        feature_names = []
        feature_values = []
        tmp_t = []
        AADict = {}

        for i in range(len(code)):
            feature_name = AA[i // 20] + AA[i % 20]
            feature_names.append(feature_name)
        tmp_t.append(feature_names)

        for i in range(len(AA)):
            AADict[AA[i]] = i

        for sequence in X:
            sequence = sequence.strip()
            # code = []
            feature_vector = []
            tmpCode = [0] * 400
            for j in range(len(sequence) - 2 + 1):
                tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 +
                                                                                      AADict[sequence[j + 1]]] + 1
            if sum(tmpCode) != 0:
                for i in tmpCode:
                    calcs = i / sum(tmpCode)
                    feature_vector.append(calcs)
                feature_values.append(feature_vector)
                tmp_t.append(feature_vector)

        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t

    # 三联体(Conjoint Triad, CT)
    def CT(x):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        code = [0] * pow(20, 3)
        feature_names = []
        feature_values = []
        tmp_t = []
        AADict = {}

        for i in range(len(code)):
            feature_name = ''.join([AA[(i >> (2 * j)) & 3] for j in range(2, -1, -1)])
            feature_names.append(feature_name)
        tmp_t.append(feature_names)

        for i in range(len(AA)):
            AADict[AA[i]] = i

        for sequence in x:
            sequence = sequence.strip()
            feature_vector = []
            tmpCode = [0] * pow(20, 3)
            for j in range(len(sequence) - 3 + 1):
                tmpCode[
                    AADict[sequence[j]] * pow(20, 2) + AADict[sequence[j + 1]] * pow(20, 1) + AADict[sequence[j + 2]]] = \
                    tmpCode[AADict[sequence[j]] * pow(20, 2) + AADict[sequence[j + 1]] * pow(20, 1) + AADict[
                        sequence[j + 2]]] + 1
            if sum(tmpCode) != 0:
                for i in tmpCode:
                    calcs = i / sum(tmpCode)
                    feature_vector.append(calcs)
                feature_values.append(feature_vector)
                tmp_t.append(feature_vector)

        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
        print()
    def getFeatures(kGap, kTuple, x,y):

        if args['length'] == 1:
            getSeqLength(x)

        if args['zCurve'] == 1:
            zCurve(x, seqType.upper())              #3

        if args['gcContent'] == 1:
            gcContent(x, seqType.upper())           #1

        if args['cumulativeSkew'] == 1:
            cumulativeSkew(x, seqType.upper())      #2

        if args['atgcRatio'] == 1:
            atgcRatio(x, seqType.upper())         #1

        if args['NAC']==1:
            NAC(x)
        # 二核苷酸组成方法(Dinucleotide Composition, DNC)
        if args['DNC']==1:
            DNC(x)
        # 三核苷酸组成(Trinucleotide composition, TNC)
        if args['TNC']==1:
            TNC(x)

        if args['kmer_freq_o'] == 1:
            kmer_freq_o(x,kTuple)

        if args['kmer_freq_t'] == 1:
            kmer_freq_t(x,kTuple)

        # if args['length'] == 1:
        #     getLength(x)

        if args['MaxORFsLen'] == 1:
            MaxORFsLen(x)

        if args['orf_coverage'] == 1:
            orf_coverage(x)

        if args['count_orfs'] == 1:
            count_orfs(x)

        if args['AAC'] == 1:
            AAC(x)

        if args['DPC'] == 1:
            DPC(x)

        if args['CT'] == 1:
            CT(x)

        tmp_t = []
        feature_name = ['label']
        tmp_t.append(feature_name)
        for i in range(len(y)):
            tmp_t.append([y[i]])
        # T.append(y)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t

    getFeatures(args['kGap'], args['kTupe'], X, Y)
    # 获取每个子列表的长度
    # lengths = [len(sublist) for sublist in T]
    # # 计算最大和最小长度
    # max_length = max(lengths)
    # min_length = min(lengths)
    # # 打印结果
    # print(f'Max length: {max_length}')
    # print(f'Min length: {min_length}')
    return np.array(T)

