from numpy import *


def createDataset():
    group = array([
        ['40-50000', 1, 0, 0, 0],
        ['30-40000', 1, 1, 0, 1],
        ['40-50000', 0, 0, 0, 0],
        ['30-40000', 1, 1, 1, 0],
        ['50-60000', 1, 0, 0, 1],
        ['20-30000', 0, 0, 0, 1],
        ['30-40000', 1, 0, 1, 0],
        ['20-30000', 0, 1, 0, 0],
        ['30-40000', 1, 0, 0, 0],
        ['30-40000', 1, 1, 0, 1],
        ['40-50000', 0, 1, 0, 1],
        ['20-30000', 0, 1, 0, 0],
        ['50-60000', 1, 1, 0, 1],
        ['40-50000', 0, 1, 0, 0],
        ['20-30000', 0, 0, 1, 1],
    ])
    labels = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1]
    return group, labels


def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print
        "the word: %s is not in my Vocabulary!" % word
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 计算某个类发生的概率
    p0Num = ones(numWords);
    p1Num = ones(numWords)  # 初始样本个数为1，防止条件概率为0，影响结果
    p0Denom = 2.0;
    p1Denom = 2.0  # 作用同上
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)  # 计算类标签为1时的其它属性发生的条件概率
    p0Vect = log(p0Num / p0Denom)  # 计算标签为0时的其它属性发生的条件概率
    return p0Vect, p1Vect, pAbusive  # 返回条件概率和类标签为1的概率


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    # step1：加载数据集和类标号
    listOPosts, listClasses = createDataset()
    # step2：创建词库
    myVocabList = createVocabList(listOPosts)
    # step3：计算每个样本在词库中的出现情况
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # step4：调用第四步函数，计算条件概率
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    # step5
    # 测试1
    testEntry = ['40-50000', 1, 0, 0, 0]
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


testingNB()
