from BayesianOptimization import BayesianOptimization
import numpy as np
import torch
from SVD import SVD

if __name__ == "__main__":
    trueFilePath = "./trans.txt"
    colNames = list()
    device = torch.device("cuda")
    with open(trueFilePath,mode="r") as rh:
        k = 0
        data = []
        for oneLine in rh:
            thisLine = oneLine.strip("\n")
            if k != 0:
                oneLineData = thisLine.split("	")[1:]
                colNames.append(thisLine.split("	")[0])
                oneLineFloatData = []
                for value in oneLineData:
                    oneLineFloatData.append(float(value))
                data.append(oneLineFloatData)
            k += 1
    matrix = np.array(data,dtype=np.float32)
    print(matrix)
    #print(matrix[-1])
    iniK = 1
    maxK = 50
    iniLambda = 0.2
    maxLambda = 0.8
    bayesOpt = BayesianOptimization(matrix,iniK,iniLambda,device=device,maxSVDTrainingTimes=70000,maxBayesianTrainingTimes=64,lr=1e-4,pValue=0.65)
    bayesOpt.train(maxK,maxLambda,sampleNumber=500)
    optim = bayesOpt.getBest()
    print(optim)
    k = optim[0]
    r = optim[1]
    ### 98.          0.27777305
    svd = SVD(matrix,int(k),device=torch.device("cuda"),biasSVD=True,prediction=True,regularization=r,trainingTimes=70000,learning_rate=1e-4)
    svd.train(verbose=True)
    finalResult = svd.prediction()
    m =  finalResult.shape[0]
    n =  finalResult.shape[1]
    with open("./predictionMatrix.txt",mode="w") as wh:
        for i in range(m):
            wh.write(colNames[i] + "\t")
            for j in range(n):
                wh.write(str(finalResult[i,j]) + "\t")
            wh.write("\n")






