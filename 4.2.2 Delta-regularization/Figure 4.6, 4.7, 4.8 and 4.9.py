import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sc
S0=1
K = 1
r = 0.06
sigma = 0.2
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
#Longstaff-Schwartz method with Delta-regularization, that gives the price
# and delta by price-only-regressing and Delta-reguarization given current
# spot-price and Time-to-maturity.
#It is a function of S0 (one-dimensional), TimeToMat (capT), Number of
#simualtions (NoSim) and degree in the polynomial
def AmrLSM(spot, capT, Nhedge, NoSim, p, NoSeed, method):
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
    STstop = np.exp(-r*dt)*St[:,Nhedge]
    ST = St[:,Nhedge]
    PayOffMatrix = (K-St)*(St<K)
    cashflowTJEK = np.exp(-r*dt)*PayOffMatrix[:,Nhedge]
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
        Xdiff = np.c_[np.zeros(len(x)), np.ones(len(x)), range(2,p+1)
                      *np.delete(X, [0,p], 1)]
        if(method == "Price-only-regression"):
            coeff = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X))
                              , np.matmul(np.transpose(X), Y))
        if(method == "Delta-regularization"):
            coeff = np.matmul(np.linalg.inv(w*np.matmul(np.transpose(X),X)
                                            + (1-w)*np.matmul(np.transpose(Xdiff), Xdiff))
                              , w*np.matmul(np.transpose(X),Y)
                              + (1-w)*np.matmul(np.transpose(Xdiff),D_in))
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
    result = {"Price": Price0, "Delta": Delta0}
    return(result)
MeanPrice = np.zeros((len(Poldeg),len(Nsim)))
StdPrice = np.zeros((len(Poldeg),len(Nsim)))
MeanPriceReg = np.zeros((len(Poldeg),len(Nsim)))
StdPriceReg = np.zeros((len(Poldeg),len(Nsim)))
MeanDelta = np.zeros((len(Poldeg),len(Nsim)))
StdDelta = np.zeros((len(Poldeg),len(Nsim)))
MeanDeltaReg = np.zeros((len(Poldeg),len(Nsim)))
StdDeltaReg = np.zeros((len(Poldeg),len(Nsim)))
for s in range(len(Nsim)):
    n = Nsim[s]
    print("NoSimulations =", n)
    PriceBatch = np.zeros((len(Poldeg), Nbatch))
    PriceBatchReg = np.zeros((len(Poldeg), Nbatch))
    DeltaBatch = np.zeros((len(Poldeg), Nbatch))
    DeltaBatchReg = np.zeros((len(Poldeg), Nbatch))
    for deg in range(len(Poldeg)):
        p1 = Poldeg[deg]
        for k in range(Nbatch):
            M = AmrLSM(spot = S0, capT = 1, Nhedge = 52, NoSim = n, p = p1, NoSeed = k
                       , method = "Delta-regularization")
            PriceBatchReg[deg, k] = M["Price"]
            DeltaBatchReg[deg, k] = M["Delta"]
            M = AmrLSM(spot = S0, capT = 1, Nhedge = 52, NoSim = n, p = p1, NoSeed = k
                       , method = "Price-only-regression")
            PriceBatch[deg, k] = M["Price"]
            DeltaBatch[deg, k] = M["Delta"]
    for deg in range(len(Poldeg)):
        MeanPrice[deg,s] = np.mean(PriceBatch[deg, :])
        StdPrice[deg,s] = np.std(PriceBatch[deg, :])
        MeanPriceReg[deg,s] = np.mean(PriceBatchReg[deg, :])
        StdPriceReg[deg,s] = np.std(PriceBatchReg[deg, :])
        MeanDelta[deg,s] = np.mean(DeltaBatch[deg, :])
        StdDelta[deg,s] = np.std(DeltaBatch[deg, :])
        MeanDeltaReg[deg,s] = np.mean(DeltaBatchReg[deg, :])
        StdDeltaReg[deg,s] = np.std(DeltaBatchReg[deg, :])
L = AmrPutBino(spot=S0, TimeToMat=1, NoTimePoints=2500)
BinPrice = L["Price"]
BinDelta = L["Delta"]
plt.subplot(1, 2, 1)
plt.plot(Nsim, MeanPrice[0,:], label = "Estimated Backward Price by Price-only-regression"
         , color = "red")
plt.text(Nsim[-1], MeanPrice[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.text(Nsim[0]-250, MeanPrice[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.plot(Nsim, MeanPriceReg[0,:], label = "Estimated Backward Price by Delta-regularization"
         , color = "blue")
plt.text(Nsim[-1], MeanPriceReg[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'blue')
plt.text(Nsim[0]-250, MeanPriceReg[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'blue')
for i in range(1,len(Poldeg)):
    plt.plot(Nsim, MeanPrice[i,:], color = "red")
    plt.text(Nsim[-1], MeanPrice[i,:][-1], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.text(Nsim[0]-250, MeanPrice[i,:][0], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.plot(Nsim, MeanPriceReg[i,:], color = "blue")
    plt.text(Nsim[-1], MeanPriceReg[i,:][-1], str(Poldeg[i]), fontsize = 10, color = 'blue')
    plt.text(Nsim[0]-250, MeanPriceReg[i,:][0], str(Poldeg[i]), fontsize = 10, color = 'blue')
plt.axhline(y=BinPrice, color='black', linestyle='-', label = "Binomial Price")
plt.xlabel('NoSimulations')
plt.ylabel('Time-zero price')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(Nsim, StdPrice[0,:], label = "Estimated standard error", color = "red")
plt.text(Nsim[-1], StdPrice[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.text(Nsim[0]-50, StdPrice[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.plot(Nsim, StdPriceReg[0,:], label = "Estimated standard error", color = "blue")
plt.text(Nsim[-1], StdPriceReg[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'blue')
plt.text(Nsim[0]-50, StdPriceReg[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'blue')
for i in range(1,len(Poldeg)):
    plt.plot(Nsim, StdPrice[i,:], color = "red")
    plt.text(Nsim[-1], StdPrice[i,:][-1], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.text(Nsim[0]-50, StdPrice[i,:][0], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.plot(Nsim, StdPriceReg[i,:], color = "blue")
    plt.text(Nsim[-1], StdPriceReg[i,:][-1], str(Poldeg[i]), fontsize = 10, color = 'blue')
    plt.text(Nsim[0]-50, StdPriceReg[i,:][0], str(Poldeg[i]), fontsize = 10, color = 'blue')
plt.axhline(y=0, color='black', linestyle='-')
plt.xlabel('NoSimulations')
plt.ylabel('Stanard error')
plt.legend()
plt.show()
plt.subplot(1, 2, 1)
plt.plot(Nsim, MeanDelta[0,:], label = "Estimated Backward Delta by Price-only-regression"
         , color = "red")
plt.text(Nsim[-1], MeanDelta[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.text(Nsim[0]-250, MeanDelta[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.plot(Nsim, MeanDeltaReg[0,:], label = "Estimated Backward Delta by Delta-regularization"
         , color = "blue")
plt.text(Nsim[-1], MeanDeltaReg[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'blue')
plt.text(Nsim[0]-250, MeanDeltaReg[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'blue')
for i in range(1,len(Poldeg)):
    plt.plot(Nsim, MeanDelta[i,:], color = "red")
    plt.text(Nsim[-1], MeanDelta[i,:][-1], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.text(Nsim[0]-250, MeanDelta[i,:][0], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.plot(Nsim, MeanDeltaReg[i,:], color = "blue")
    plt.text(Nsim[-1], MeanDeltaReg[i,:][-1], str(Poldeg[i]), fontsize = 10, color = 'blue')
    plt.text(Nsim[0]-250, MeanDeltaReg[i,:][0], str(Poldeg[i]), fontsize = 10, color = 'blue')
plt.axhline(y=BinDelta, color='black', linestyle='-', label = "Binomial Delta")
plt.xlabel('NoSimulations')
plt.ylabel('Time-zero Delta')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(Nsim, StdDelta[0,:], label = "Estimated standard error", color = "red")
plt.text(Nsim[-1], StdDelta[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.text(Nsim[0]-50, StdDelta[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.plot(Nsim, StdDeltaReg[0,:], label = "Estimated standard error", color = "blue")
plt.text(Nsim[-1], StdDeltaReg[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'blue')
plt.text(Nsim[0]-50, StdDeltaReg[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'blue')
for i in range(1,len(Poldeg)):
    plt.plot(Nsim, StdDelta[i,:], color = "red")
    plt.text(Nsim[-1], StdDelta[i,:][-1], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.text(Nsim[0]-50, StdDelta[i,:][0], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.plot(Nsim, StdDeltaReg[i,:], color = "blue")
    plt.text(Nsim[-1], StdDeltaReg[i,:][-1], str(Poldeg[i]), fontsize = 10, color = 'blue')
    plt.text(Nsim[0]-50, StdDeltaReg[i,:][0], str(Poldeg[i]), fontsize = 10, color = 'blue')
plt.axhline(y=0, color='black', linestyle='-')
plt.xlabel('NoSimulations')
plt.ylabel('Stanard error')
plt.legend()
plt.show()
def PriceConfInterval(degree):
    upperboundReg = MeanPriceReg - sc.norm.ppf(0.975)*StdPrice
    lowerboundReg = MeanPriceReg + sc.norm.ppf(0.975)*StdPrice
    upperboundDeg = MeanPrice - sc.norm.ppf(0.975)*StdPrice
    lowerboundDeg = MeanPrice + sc.norm.ppf(0.975)*StdPrice
    plt.plot()
    plt.plot(Nsim, MeanPrice[degree,:], label = "Estimated Backward price by price-only-regression"
             , color = "red")
    plt.text(Nsim[-1], MeanPrice[degree,:][-1], str(Poldeg[degree]), fontsize = 10
             , color = 'red')
    plt.plot(Nsim, upperboundDeg[degree,:], label = "95% confidence interval"
             , color = "red", linestyle = "dashed")
    plt.plot(Nsim, lowerboundDeg[degree,:],color = "red",  linestyle = "dashed")
    plt.plot(Nsim, MeanPriceReg[degree,:], label = "Estimated Backward price by Delta-regularization"
             , color = "blue")
    plt.text(Nsim[-1], MeanPriceReg[degree,:][-1], str(Poldeg[degree]), fontsize = 10
             , color = 'blue')
    plt.plot(Nsim, upperboundReg[degree,:], label = "95% confidence interval"
             , color = "blue", linestyle = "dashed")
    plt.plot(Nsim, lowerboundReg[degree,:],color = "blue",  linestyle = "dashed")
    plt.axhline(y=BinPrice, color='black', linestyle='-', label = "Binomial Price")
    plt.xlabel('NoSimulations')
    plt.ylabel('Time-zero Price')
    plt.title("American put-Price by " + str(Poldeg[degree])
              +"-degree polynomial fit given S(0)=" + str(S0))
    plt.legend()
    plt.show()
def DeltaConfInterval(degree):
    upperboundReg = MeanDeltaReg - sc.norm.ppf(0.975)*StdDelta
    lowerboundReg = MeanDeltaReg + sc.norm.ppf(0.975)*StdDelta
    upperboundDeg = MeanDelta - sc.norm.ppf(0.975)*StdDelta
    lowerboundDeg = MeanDelta + sc.norm.ppf(0.975)*StdDelta
    plt.plot()
    plt.plot(Nsim, MeanDelta[degree,:], label = "Estimated Backward Delta by price-only-regression", color = "red")
    plt.text(Nsim[-1], MeanDelta[degree,:][-1], str(Poldeg[degree]), fontsize = 10
             , color = 'red')
    plt.plot(Nsim, upperboundDeg[degree,:], label = "95% confidence interval"
             , color = "red", linestyle = "dashed")
    plt.plot(Nsim, lowerboundDeg[degree,:],color = "red",  linestyle = "dashed")
    plt.plot(Nsim, MeanDeltaReg[degree,:], label = "Estimated Backward Delta by Delta-regularization", color = "blue")
    plt.text(Nsim[-1], MeanDeltaReg[degree,:][-1], str(Poldeg[degree]), fontsize = 10
             , color = 'blue')
    plt.plot(Nsim, upperboundReg[degree,:], label = "95% confidence interval"
             , color = "blue", linestyle = "dashed")
    plt.plot(Nsim, lowerboundReg[degree,:],color = "blue",  linestyle = "dashed")
    plt.axhline(y=BinDelta, color='black', linestyle='-', label = "Binomial Delta")
    plt.xlabel('NoSimulations')
    plt.ylabel('Time-zero Delta')
    plt.title("American put-Delta by " + str(Poldeg[degree])
              +"-degree polynomial fit given S(0)=" + str(S0))
    plt.legend()
    plt.show()
