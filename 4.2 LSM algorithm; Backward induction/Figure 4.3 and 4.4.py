import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sc
K = 1
r = 0.06
sigma = 0.2
S0 = 1
capT = 1
Nhedge = 52
dt = capT/Nhedge
Poldeg = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
Nbatch = 250
Nsim = np.array([500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
def AmrPutBino(spot, TimeToMat, NoTimePoints):
    S0 = spot
    dt = TimeToMat/NoTimePoints
    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))
    R = np.exp(r*dt)
    q = (R-d)/(u-d)
    St = np.ones((NoTimePoints+1,NoTimePoints+1))
    St[0][0] = S0
    for k in range(NoTimePoints+1):
        if k>0:
            St[k][k] = d*St[k-1][k-1]
        if k<(NoTimePoints):
            for j in range(k,NoTimePoints):
                St[k][j+1] = u*St[k][j]
    payoff = (K-St)*(St<K)
    cashflow = np.zeros((NoTimePoints+1,NoTimePoints+1))
    cashflow[:,NoTimePoints] = payoff[:,NoTimePoints]
    for i in range(NoTimePoints):
        l = NoTimePoints-i
        for j in range(l):
            exVal = payoff[j][l-1]
            contVal = (1/R)*(q*cashflow[j][l] + (1-q)*cashflow[j+1][l])
            cashflow[j][l-1] = exVal*(exVal>= contVal) + contVal*(exVal<contVal)
    
    Delta = (cashflow[0][1]-cashflow[1][1])/(St[0][1]-St[1][1])
    result = {"Price": cashflow[0][0], "Delta": Delta}
    return(result)
def AmrLSM(spot, capT, Nhedge, NoSim, p, NoSeed):
    S0 = spot
    n = NoSim
    dt = capT/Nhedge
    w = 0.5
    np.random.seed(seed=NoSeed+1)
    St = np.zeros((n, Nhedge +1))
    St[:,0] = S0
    for i in range(Nhedge):
        Z = np.random.normal(0, 1, size=n)
        St[:,i+1] = St[:,i]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    PayOffMatrix = (K-St)*(St<K)
    cashflowTJEK = np.exp(-r*dt)*PayOffMatrix[:,Nhedge]
    for i in range(Nhedge-1):
        q = Nhedge -1 -i
        inMonPosition = np.where(PayOffMatrix[:,q] !=0)
        x = St[inMonPosition, q]
        x = x[0]
        Y = cashflowTJEK[inMonPosition]
        X = np.vander(x-K, p+1, increasing=True)
        coeff = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X))
                          , np.matmul(np.transpose(X), Y))
        contVal = np.matmul(X, coeff)
        exVal = PayOffMatrix[inMonPosition,q]
        cashflowTJEK[inMonPosition] = exVal*(exVal>=contVal)*1 + Y*(exVal<contVal)
        cashflowTJEK = np.exp(-r*dt)*cashflowTJEK
    Price0 = np.mean(cashflowTJEK)
    result = {"PriceBackward": Price0}
    return(result)
MeanPrice = np.zeros((len(Poldeg),len(Nsim)))
StdPrice = np.zeros((len(Poldeg),len(Nsim)))
StdPriceApprox = np.zeros((len(Poldeg),len(Nsim)))
for s in range(len(Nsim)):
    n = Nsim[s]
    print("NoSimulations =", n)
    PriceBatch = np.zeros((len(Poldeg), Nbatch))
    for deg in range(len(Poldeg)):
        p1 = Poldeg[deg]
        for k in range(Nbatch):
            M = AmrLSM(spot = S0, capT = 1, Nhedge = 52, NoSim = n, p = p1, NoSeed = k)
            PriceBatch[deg, k] = M["PriceBackward"]
    for deg in range(len(Poldeg)):
        MeanPrice[deg,s] = np.mean(PriceBatch[deg, :])
        StdPrice[deg,s] = np.std(PriceBatch[deg, :])
        StdPriceApprox[deg,s] = StdPrice[deg,s]*np.sqrt(Nbatch)/np.sqrt(n)
L = AmrPutBino(spot=S0, TimeToMat=1, NoTimePoints=750)
BinPrice = L["Price"]
plt.subplot(1, 2, 1)
plt.plot(Nsim, MeanPrice[0,:], label = "Approximated Backward price", color = "red")
plt.text(Nsim[-1], MeanPrice[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.text(Nsim[0]-250, MeanPrice[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'red')
for i in range(1,len(Poldeg)):
    plt.plot(Nsim, MeanPrice[i,:], color = "red")
    plt.text(Nsim[-1], MeanPrice[i,:][-1], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.text(Nsim[0]-250, MeanPrice[i,:][0], str(Poldeg[i]), fontsize = 10, color = 'red')
plt.axhline(y=BinPrice, color='black', linestyle='-', label = "Binomial price")
plt.xlabel('NoSimulations')
plt.ylabel('Time-zero price')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(Nsim, StdPrice[0,:], label = "Estimated standard error", color = "red")
plt.text(Nsim[-1], StdPrice[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.text(Nsim[0]-50, StdPrice[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'red')
for i in range(1,len(Poldeg)):
    plt.plot(Nsim, StdPrice[i,:], color = "red")
    plt.text(Nsim[-1], StdPrice[i,:][-1], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.text(Nsim[0]-50, StdPrice[i,:][0], str(Poldeg[i]), fontsize = 10, color = 'red')
plt.axhline(y=0, color='black', linestyle='-')
plt.xlabel('NoSimulations')
plt.ylabel('Standard error')
plt.legend()
plt.show()
upperbound = MeanPrice - sc.norm.ppf(0.975)*StdPrice
lowerbound = MeanPrice + sc.norm.ppf(0.975)*StdPrice
def ConfInterval(degree):
    plt.plot()
    plt.plot(Nsim, MeanPrice[degree,:], label = "Approximated Backward price", color = "red")
    plt.text(Nsim[-1], MeanPrice[degree,:][-1], str(Poldeg[degree]), fontsize = 10, color = 'red')
    plt.plot(Nsim, upperbound[degree,:], label = "95% confidence interval", color = "red", linestyle = "dashed")
    plt.plot(Nsim, lowerbound[degree,:],color = "red",  linestyle = "dashed")
    plt.axhline(y=BinPrice, color='black', linestyle='-', label = "Binomial price")
    plt.xlabel('NoSimulations')
    plt.ylabel('Time-zero price')
    plt.title("American put-price by " + str(Poldeg[degree]) + "-degree polynomial fit given S(0)=" + str(S0))
    plt.legend()
    plt.show()
