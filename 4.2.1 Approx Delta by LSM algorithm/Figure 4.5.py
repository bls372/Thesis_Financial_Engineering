import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sc
K = 1
r = 0.06
sigma = 0.2
Poldeg = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
Nbatch = 250
Nsim = np.array([500,1000, 2000,3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
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
    np.random.seed(seed=NoSeed+1)
    St = np.zeros((n, Nhedge +1))
    St[:,0] = S0
    for i in range(Nhedge):
        Z = np.random.normal(0, 1, size=n)
        St[:,i+1] = St[:,i]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    STstop = np.exp(-r*dt)*St[:,Nhedge]
    ST = St[:,Nhedge]
    PayOffMatrix = (K-St)*(St<K)
    cashflowTJEK = np.exp(-r*dt)*PayOffMatrix[:,Nhedge]
    coeffs = np.zeros((Nhedge, p+1))
    for i in range(Nhedge-1):
        q = Nhedge -1 -i
        inMonPosition = np.where(PayOffMatrix[:,q] !=0)
        x = St[inMonPosition, q]
        x = x[0]
        Y = cashflowTJEK[inMonPosition]
        X = np.vander(x-K, p+1, increasing=True)
        ST_in = ST[inMonPosition]
        discST_in = STstop[inMonPosition]
        D_in = -1*(ST_in<K)*discST_in/x
        coeff = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X))
                          , np.matmul(np.transpose(X), Y))
        contVal = np.matmul(X, coeff)
        exVal = PayOffMatrix[inMonPosition,q]
        cashflowTJEK[inMonPosition] = exVal*(exVal>=contVal)*1 + Y*(exVal<contVal)
        STstop[inMonPosition] = x*(exVal>=contVal)*1+ discST_in*(exVal<contVal)
        ST[inMonPosition] = x*(exVal>=contVal)*1 + ST_in*(exVal<contVal)
        cashflowTJEK = np.exp(-r*dt)*cashflowTJEK
        STstop = np.exp(-r*dt)*STstop
    Price0 = np.mean(cashflowTJEK)
    discST = STstop
    D = -1*(ST<K)*discST/S0
    Delta0 = np.mean(D)
    result = {"PriceBackward": Price0, "DeltaBackward": Delta0}
    return(result)
MeanDelta = np.zeros((len(Poldeg),len(Nsim)))
StdDelta = np.zeros((len(Poldeg),len(Nsim)))
for s in range(len(Nsim)):
    n = Nsim[s]
    print("NoSimulations =", n)
    DeltaBatch = np.zeros((len(Poldeg), Nbatch))
    for deg in range(len(Poldeg)):
        p1 = Poldeg[deg]
        for k in range(Nbatch):
            LSM = AmrLSM(spot = 1, capT = 1, Nhedge = 52, NoSim = n, p = p1, NoSeed = k)
            DeltaBatch[deg, k] = LSM["DeltaBackward"]
    for deg in range(len(Poldeg)):
        MeanDelta[deg,s] = np.mean(DeltaBatch[deg, :])
        StdDelta[deg,s] = np.std(DeltaBatch[deg, :])

L = AmrPutBino(spot=1, TimeToMat=1, NoTimePoints=750)
BinPrice = L["Delta"]
plt.subplot(1, 2, 1)
plt.plot(Nsim, MeanDelta[0,:], label = "Approximated Backward Delta", color = "red")
plt.text(Nsim[-1], MeanDelta[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.text(Nsim[0]-250, MeanDelta[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'red')
for i in range(1,len(Poldeg)):
    plt.plot(Nsim, MeanDelta[i,:], color = "red")
    plt.text(Nsim[-1], MeanDelta[i,:][-1], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.text(Nsim[0]-250, MeanDelta[i,:][0], str(Poldeg[i]), fontsize = 10, color = 'red')
plt.axhline(y=BinPrice, color='black', linestyle='-', label = "Binomial Delta")
plt.xlabel('NoSimulations')
plt.ylabel('Time-zero Delta')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(Nsim, StdDelta[0,:], label = "Estimated standard error", color = "red")
plt.text(Nsim[-1], StdDelta[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.text(Nsim[0]-50, StdDelta[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'red')
for i in range(1,len(Poldeg)):
    plt.plot(Nsim, StdDelta[i,:], color = "red")
    plt.text(Nsim[-1], StdDelta[i,:][-1], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.text(Nsim[0]-50, StdDelta[i,:][0], str(Poldeg[i]), fontsize = 10, color = 'red')
plt.axhline(y=0, color='black', linestyle='-')
plt.xlabel('NoSimulations')
plt.ylabel('Standard error')
plt.legend()
plt.show()
upperbound = MeanDelta - sc.norm.ppf(0.975)*StdDelta
lowerbound = MeanDelta + sc.norm.ppf(0.975)*StdDelta
def ConfInterval(degree):
    plt.plot()
    plt.plot(Nsim, MeanDelta[degree,:], label = "Approximated Backward Delta", color = "red")
    plt.text(Nsim[-1], MeanDelta[degree,:][-1], str(Poldeg[degree]), fontsize = 10
             , color = 'red')
    plt.plot(Nsim, upperbound[degree,:], label = "95% confidence interval", color = "red"
             , linestyle = "dashed")
    plt.plot(Nsim, lowerbound[degree,:],color = "red",  linestyle = "dashed")
    plt.axhline(y=BinPrice, color='black', linestyle='-', label = "Binomial Delta")
    plt.xlabel('NoSimulations')
    plt.ylabel('Time-zero Delta')
    plt.title("American put-Delta by " + str(Poldeg[degree])
              +"-degree polynomial fit given S(0)=" + str(1))
    plt.legend()
    plt.show()
        
