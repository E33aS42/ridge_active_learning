import numpy as np
import sys
import copy

"""
This script should be launched from the console using the following line:
python reg.py lambda sigma2 X_train y_train X_test
Where:
. regul_reg.py is the name of this python file
. lambda is a regularization parameter.
. sigma2 is the variance.

. X_train is a .csv file of the data nodes coordinates.
. y_train is a .csv file that should contain the known labels corresponding 
to the nodes in the X_train file
. X_test is a .csv file of the data nodes that will be predicted after training
"""


## Preprocessing
# The data has to be standardized to avoid too much disparities (small and big values) 
# between data points and avoiding as scuh each output to be penalized differently.

def standardize(X):
    shaX = np.shape(X)
    x = X[:]
    
    # calculate global mean for each dimension
    meanXj = np.zeros((shaX[1], 1))
    for j in range(shaX[1]):
        for i in range(shaX[0]):
             meanXj[j]+= X[i][j]
        meanXj[j] = meanXj[j]/shaX[0]
    
    # calculate variance for each dimension
    sigmaj = np.zeros((shaX[1], 1))
    for j in range(shaX[1]):
        sum = 0
        for i in range(shaX[0]):
            sum += (X[i][j] - meanXj[j])**2
        sum = sum/shaX[0]
        sigmaj[j] = np.sqrt(sum)
    
    ## standardize dimensions of xi
    for j in range(shaX[1]-1):
        for i in range(shaX[0]):
            x[i][j] = (x[i][j]-meanXj[j])/sigmaj[j]
    
    return x

def preprocessing(X_train, y_train, X_test):
    lenY = len(y_train)
    y = y_train[:]
    x = standardize(X_train)
    x_0 = standardize(X_test)
    
    ## global mean substracted off y
    for i in range(lenY):
        y[i] -= np.mean(y_train)
    
    ## dimensions of xi are standardized 
    x = standardize(X_train)
    x_0 = standardize(X_test)

    return x, y, x_0

## Ridge regression parameter
def RR(X, y, lamb):
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    shax = np.shape(X)
    prod1 = lamb * np.identity(shax[1])
    prod2 = np.dot(np.transpose(X), X)
    prod3 = np.dot(np.transpose(X), y)
    print(np.shape(prod1), np.shape(prod2), np.shape(prod3))
    inv1 = np.linalg.inv(prod1 + prod2)
    wRR = np.dot(inv1, prod3)
    print(wRR)
    return wRR

## Lasso regression
def Lasso(X, y, lamb):
    shax = np.shape(X)
    prod1 = lamb * np.identity(shax[1])
    prod2 = np.dot(np.transpose(X), X)
    prod3 = np.dot(np.transpose(X), y)
    print(np.shape(prod1), np.shape(prod2), np.shape(prod3))
    inv1 = np.linalg.inv(prod2)
    wL = np.dot(inv1, prod3 - 1/2*prod1)
    print(wL)
    return wL

## Elastic net
def El_net(X, y, lamb1, lamb2):
    shax = np.shape(X)
    prod1a = lamb1 * np.identity(shax[1])
    prod1b = lamb2 * np.identity(shax[1])
    prod2 = np.dot(np.transpose(X), X)
    prod3 = np.dot(np.transpose(X), y)
    print(np.shape(prod1a), np.shape(prod1b), np.shape(prod2), np.shape(prod3))
    inv1 = np.linalg.inv(prod1b + prod2)
    wEn = np.dot(inv1, prod3 - 1/2*prod1a)
    print(wEn)
    return wEn 

## Gaussian distribution
def gaussian(X, mu, sig):
    return np.exp(-np.power(X - mu, 2.) / (2 * np.power(sig, 2.)))

## Active learning
def AL(X, y, lamb, sigm, X_0):
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    shax = np.shape(X)
    prod1 = lamb * np.identity(shax[1])
    prod2 = 1/sigm*np.dot(np.transpose(X), X)
    Sigma = np.linalg.inv(prod1 + prod2)
    # print('Sigma = %s'% Sigma)
    inv1 = np.linalg.inv(sigm*prod1 + sigm*prod2)
    prod3 = np.dot(np.transpose(X), y)
    mu = np.dot(inv1,prod3)
    # print('mu = %s'% mu)
    print('size(X_0) = {}'.format(np.shape(X_0)))
    print('size(Sigma) = {}'.format(np.shape(Sigma)))
    prod4 = np.dot(Sigma, np.transpose(X_0))
    # print('prod4 = {}'.format(prod4))
    sigm_0 = sigm + np.dot(X_0, prod4)
    # print('sigm_0 = %s'% sigm_0)

    #Sigma_0 = np.linalg.inv(np.linalg.inv(Sigma) + 1/sigm*np.dot(np.transpose(X_0), X_0))
    # print('Sigma_0 = %s' % Sigma_0)

    mu_0 = np.dot(X_0,mu)
    # print('mu_0 = %s'% mu_0)
    shasig = np.shape(sigm_0)
    sigma0 = np.zeros((shasig[0],1))
    for i in range(shasig[0]):
        sigma0[i] = sigm_0[i][i]

    print(sigm_0)
    print(sigma0)
    index = np.zeros(10)
    count1 = shasig[0]
    count2 = 0
    sig0 = copy.deepcopy(sigma0)
    sig0 = sig0.tolist()
    print(sig0)
    while count1 > shasig[0]-10:
        sigmax = max(sig0)
        i = 1
        for val in sigma0:
            if val == sigmax:
                index[count2] = i
                sig0.remove(sigmax)
                break
            else:
                i += 1

        count1 -= 1
        count2 += 1

    return index

if __name__ == "__main__":
    lambda_input = int(sys.argv[1])
    sigma2_input = float(sys.argv[2])
    X_train = np.genfromtxt(sys.argv[3], delimiter=",")
    y_train = np.genfromtxt(sys.argv[4])
    X_test = np.genfromtxt(sys.argv[5], delimiter=",")
    x, y, x_0 = preprocessing(X_train, y_train, X_test)
    
    # Ridge regression
    wRR = RR(x, y, lambda_input)  # Assuming wRR is returned from the function
    np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file
    
    # Active learning
    active = AL(x, y, lambda_input, sigma2_input, x_0)  # Assuming active is returned from the function
    np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file