import numpy as np
from SVD import SVD
import sklearn.gaussian_process as gp
from scipy.stats import norm
import torch


class BayesianOptimization(object):

    def __init__(self,matrix,iniK, iniLambda,device = torch.device("cpu"),biasSVD = True,lr = 1e-3,pValue = 0.75,
                 maxSVDTrainingTimes = 50000,maxBayesianTrainingTimes = 150):

        ### if there are zeros in matrix, the position which value is zero will be predicted.
        ### we need to add a small number to prevent this matrix has zero values.
        self.matrix = np.array(matrix,dtype=np.float32) + 1e-4
        self.m = self.matrix.shape[0]
        self.n = self.matrix.shape[1]
        self.lr = lr
        self.p = pValue
        rand = np.random.rand(self.m, self.n)
        ### 80% as training set and 20% as testing set.
        self.oneMask = np.array(rand >= 0.2, dtype= np.float32)
        self.matrix = self.matrix * self.oneMask
        self.gp_kernel = gp.kernels.RBF() + gp.kernels.WhiteKernel(1e-1)
        self.model = gp.GaussianProcessRegressor(kernel=self.gp_kernel)
        self.k = iniK
        self.lambdaR = iniLambda
        self.maxSVDTrainingTimes = maxSVDTrainingTimes
        self.biasSVD = biasSVD
        self.svd = SVD(self.matrix,self.k,biasSVD=biasSVD,prediction=True,
                       regularization=self.lambdaR,
                       trainingTimes=maxSVDTrainingTimes,device=device,learning_rate=lr)
        self.device = device
        self.trainingTimes = maxBayesianTrainingTimes
        self.X = []
        self.y = []
        self.est = []
        self.predict = []


    def maxAcquisition(self,Xnew):
        Xnew = np.array(Xnew)
        fMaxT_1 = np.max(self.y)
        yHatMean, std = self.model.predict(Xnew,return_std=True)
        # print(yHatMean)
        # print(std)
        mu = np.reshape(yHatMean,[len(yHatMean)])
        scores = (mu - fMaxT_1) * norm.cdf((mu - fMaxT_1) / (std + 1e-9)) + std * norm.pdf((mu - fMaxT_1) / (std + 1e-9))
        ix = np.argmax(scores)
        self.predict.append(mu[ix])
        return Xnew[ix,:]

    def estimator_Calculate(self):
        print("Train Current SVD.")
        self.svd.train(verbose=True)
        zeroMask = np.array(self.matrix == 0, dtype=np.float32)
        mse = np.sum(np.abs(self.matrix - self.svd.prediction()) * zeroMask)
        print("Estimator is : ", 10000. / (mse + 1e-9))
        self.est.append(mse)
        return 10000. / (mse + 1e-9)

    @staticmethod
    def randomSearchK(minBoundary, maxBoundary):
        k = int((maxBoundary - minBoundary) * np.random.rand() + minBoundary)
        return k

    @staticmethod
    def randomSearchLambda(minBoundary, maxBoundary):
        l = (maxBoundary - minBoundary) * np.random.rand() + minBoundary
        return l


    def train(self,maxK,maxLambda,sampleNumber = 200):
        """
        :param maxK: must larger than 1
        :param maxLambda: must larger than 0
        :param sampleNumber : sample number in one sampling
        :return:
        """
        print("Bayesian training start.")
        self.X.append([self.k, self.lambdaR])
        self.y.append(self.estimator_Calculate())
        self.model = self.model.fit(X=np.array(self.X),y = np.array(self.y))
        for t in range(self.trainingTimes):
            print("#################")
            print("It is at " + str(t) + " training.")
            Xsamples = [[self.randomSearchK(1,maxK), self.randomSearchLambda(0., maxLambda)] for _ in range(sampleNumber)]
            #print(Xsamples)
            p = np.random.rand(1)
            print("P value : ",p)
            if p <= self.p:
                nextX = self.maxAcquisition(Xsamples)
                self.k = nextX[0]
                print("The next k value is : ", self.k)
                self.lambdaR = nextX[1]
                print("The next lambda value is : ", self.lambdaR)
            else:
                index = int(np.random.rand(1) * len(Xsamples))
                nextX = Xsamples[index]
                self.k = nextX[0]
                print("The next k value is : ", self.k)
                self.lambdaR = nextX[1]
                print("The next lambda value is : ", self.lambdaR)
            self.svd = SVD(self.matrix, int(self.k), biasSVD=self.biasSVD, prediction=True,
                           regularization=self.lambdaR,
                           trainingTimes=self.maxSVDTrainingTimes,device=self.device,learning_rate=self.lr)
            actual = self.estimator_Calculate()
            self.X.append([self.k, self.lambdaR])
            self.y.append(actual)
            print(np.array(self.X))
            print(np.array(self.y))
            self.model = self.model.fit(np.array(self.X),y=np.array(self.y))

    def returnInfor(self):
        return self.X, self.y, self.est, self.predict,

    def getBest(self):
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        ix = np.argmax(self.y)
        return self.X[ix,:]


if __name__ == "__main__":
    testData = np.array([[1, 2, 2, 3, 2, 6], [1, 2, 7, 4, 5, 7], [1, 6, 2, 8, 5, 6], [1, 2, 3, 4, 5, 8]],
                        dtype=np.float32)
    testBayse = BayesianOptimization(testData,iniK=8,iniLambda=0.5,maxSVDTrainingTimes=10000,maxBayesianTrainingTimes=5,device=torch.device("cuda"))
    testBayse.train(maxK=50,maxLambda=1.0)
    print(testBayse.getBest())






