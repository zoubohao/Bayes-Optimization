import torch
import torch.nn as nn
import numpy as np
import torch.optim.adam as adam




class SVD (object):

    def __init__(self,matrix, k, device, biasSVD = False,prediction = True,regularization = 1e-3,learning_rate = 1e-4,trainingTimes = 500):
        """
    The Q and P are the matrices which we should optimize.
    When the training has completed, you can get the Q matrix and P matrix.
    The dimension of original matrix is [M,N],
    The dimension of Q is [M,k]
    The dimension of P is [k,N]
    The elements in matrix must equal or larger than zero. The elements which are equaled with zero need to predict.
        :param matrix: Data matrix. shape : [M,N]
        :param k: The number of latent variables
        :param biasSVD: If use Bias SVD algorithm. Default is False.
        :param regularization: The regularization of parameters.
        :param learning_rate: learning rate when using gradient descent algorithm.
        :param trainingTimes: Max training times.
        :param prediction: if use prediction model. If false, the all elements in the matrix will be calculated loss. If true, the elements which equal
        with zero are not calculated loss.
        """
        super(SVD,self).__init__()
        torch.manual_seed(seed=1024)
        self.lr = learning_rate
        self.reg = regularization
        self.device = device
        self._k = k
        self._trainingTimes = trainingTimes
        self._ifBiasSVD = biasSVD
        self._matrix = torch.from_numpy(matrix).to(device)
        self._M , self._N = self._matrix.shape[0] , self._matrix.shape[1]
        judgeMatrix = (self._matrix >= 0 ).float().to(device)
        assert  torch.sum(judgeMatrix).cpu().detach().numpy() == (self._M * self._N * 1. ) , "There exist elements which are smaller than zero."
        self._Q = torch.nn.init.xavier_normal_(nn.Parameter(torch.randn(size=[self._M,self._k]
                                          ,requires_grad=True,device=device),requires_grad=True))
        self._P = torch.nn.init.xavier_normal_(nn.Parameter(torch.randn(size=[self._k,self._N]
                                          ,requires_grad=True,device=device),requires_grad=True))
        self.parametersList = [self._Q,self._P]
        self._mask = torch.ones(size=[self._M, self._N], dtype=torch.float32).to(device)
        if prediction:
            self._mask =torch.where(self._matrix == 0, self._matrix, self._mask).to(device)

        if biasSVD:
            self._mean = np.mean(matrix)
            self._BiasM = nn.Parameter(torch.randn(size=[self._M,1],requires_grad=True,device=device)
                                   ,requires_grad=True)
            self._BiasN = nn.Parameter(torch.randn(size=[1,self._N],requires_grad=True,device=device)
                                   ,requires_grad=True)
            self.parametersList.append(self._BiasN)
            self.parametersList.append(self._BiasM)


    def _LossCalculate(self) :
        if self._ifBiasSVD:
            loss = torch.sum(torch.pow(self._matrix - self._BiasN - self._BiasM -
                                       torch.matmul(torch.abs(self._Q),torch.abs(self._P)) - self._mean, 2.0) * self._mask)
        else:
            loss = torch.sum(torch.pow(self._matrix - torch.matmul(torch.abs(self._Q),torch.abs(self._P)),2.0) * self._mask)
        return loss


    def train(self,verbose = True):
        optimizer = adam.Adam(self.parametersList, lr=self.lr, weight_decay=self.reg)
        for t in range(self._trainingTimes):
            loss = self._LossCalculate()
            if verbose :
                if t % 5000 == 0:
                    print(t)
                    print("Loss is : ", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            print("Training has completed.")


    def getQMatrix(self):
        return torch.abs(self._Q).cpu().detach().numpy()

    def getPMatrix(self):
        return torch.abs(self._P).cpu().detach().numpy()

    def prediction(self):
        """
        If there exist elements smaller than zero, it will be changed to zero.
        :return: prediction matrix.
        """
        if self._ifBiasSVD:
            predictionTensor = torch.matmul(torch.abs(self._Q), torch.abs(self._P)) + self._BiasN + self._BiasM + self._mean
        else:
            predictionTensor = torch.matmul(torch.abs(self._Q),torch.abs(self._P))
        mask = (predictionTensor >= 0.).float()
        return (predictionTensor * mask).cpu().detach().numpy()

if __name__ == "__main__":
    testData = np.array([[1,2,2,0,2,6],[1,2,0,4,5,7],[1,6,0,8,5,6],[1,0,3,4,5,0]],dtype=np.float32)
    testSVD = SVD(testData,k=10,device=torch.device("cuda"),biasSVD=True,prediction=False,regularization=0.1,trainingTimes=40000,learning_rate=0.0001)
    testSVD.train()
    print(testData)
    print(testSVD.prediction())







