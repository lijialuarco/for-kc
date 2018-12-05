from numpy import *


class KNN:
    def createDataset(self):
        group = array([[0.0, 0.2, 0.8], [9.2, 0.7, 1.5], [4.9, 0.1, 2.9], [2.7, 5.3, 6.2], [2.4, 0.0, 3.7]])
        labels = ['A', 'B', 'A', 'B', 'A']
        return group, labels

    def KnnClassify(self, testX, trainX, labels, K):
        [N, M] = trainX.shape

        # Euclidean
        difference = tile(testX, (N, 1)) - trainX  # tile for array and repeat for matrix
        difference = difference ** 2  # take pow(difference,2)
        distance = difference.sum(1)  # take the sum of difference from all dimensions
        distance = distance ** 0.5
        sortdiffidx = distance.argsort()

        # find the k nearest neighbours
        vote = {}
        for i in range(K):
            ith_label = labels[sortdiffidx[i]]
            vote[ith_label] = vote.get(ith_label,
                                       0) + 1
        sortedvote = sorted(vote.items(), key=lambda x: x[1], reverse=True)
        return sortedvote[0][0]


k = KNN()  # create KNN object
group, labels = k.createDataset()
cls = k.KnnClassify([6.3, 5.1, 0.4], group, labels, 3)
print(cls)
