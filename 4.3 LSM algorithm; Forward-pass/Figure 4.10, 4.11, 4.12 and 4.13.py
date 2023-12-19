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
NbatchForward = 250
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
        Xdiff = np.c_[np.zeros(len(x)), np.ones(len(x)), range(2,p+1)
                      *np.delete(X, [0,p], 1)]
        if(method == "Price-only-regression"):
            coeff = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X))
                              , np.matmul(np.transpose(X), Y))
            coeffs[q,:] = coeff
        if(method == "Delta-regularization"):
            coeff = np.matmul(np.linalg.inv(w*np.matmul(np.transpose(X),X)
                                            +(1-w)*np.matmul(np.transpose(Xdiff), Xdiff))
                              , w*np.matmul(np.transpose(X),Y)
                              +(1-w)*np.matmul(np.transpose(Xdiff),D_in))
            coeffs[q,:] = coeff
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
    result = {"PriceBackward": Price0, "DeltaBackward": Delta0, "Coefficients": coeffs}
    return(result)
def AmrLSMforward(spot, capT, Nhedge, NoSim, p, NoSeed, coefficients):
    S0 = spot
    n = NoSim
    dt = capT/Nhedge
    np.random.seed(seed=NoSeed+10000)
    St = np.zeros((n, Nhedge+1))
    St[:,0] = S0
    StopRule = np.zeros((n, Nhedge+1))
    coeffs = coefficients
    for i in range(Nhedge-1):
        Z = np.random.normal(0, 1, size=n)
        St[:,i+1] = St[:,i]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
        payoffs = (K-St[:,i+1])*(St[:,i+1]<K)
        inMonPosition = np.where(payoffs !=0)
        x = St[inMonPosition, i+1]
        x = x[0]
        X = np.vander(x-K, p+1, increasing=True)
        exVal = payoffs[inMonPosition]
        coeff = coeffs[i+1,:]
        contVal = np.matmul(X, coeff)
        StopRule[inMonPosition, i+1] = 1*(exVal>=contVal)
    StopRule[:, Nhedge] = 1
    for j in range(2, Nhedge+1):
        StopRule[:,range(j, Nhedge+1)] = (StopRule[:,range(j, Nhedge+1)].T
                                          *(StopRule[:,j-1] == 0)).T
    Z = np.random.normal(0, 1, size=n)
    St[:,Nhedge] = St[:,Nhedge-1]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    PayoffMatrix = (K-St)*(St<K)
    DiscPayoffMatrix = np.exp(-r*dt*np.array(range(Nhedge+1)))*PayoffMatrix*StopRule
    DiscPayoffVector = DiscPayoffMatrix.sum(axis=1)
    priceForward = np.mean(DiscPayoffVector)
    StMatrix = St*StopRule
    DiscStMatrix = np.exp(-r*dt*np.array(range(Nhedge+1)))*StMatrix
    STs = StMatrix.sum(axis=1)
    DiscSTs = DiscStMatrix.sum(axis=1)
    D = -1*(STs<K)*DiscSTs/S0
    DeltaForward = np.mean(D)
    result = {"PriceForward": priceForward, "DeltaForward": DeltaForward}
    return(result)
MeanPrice = np.zeros((len(Poldeg),len(Nsim)))
StdPrice = np.zeros((len(Poldeg),len(Nsim)))
MeanPriceReg = np.zeros((len(Poldeg),len(Nsim)))
StdPriceReg = np.zeros((len(Poldeg),len(Nsim)))
MeanDelta = np.zeros((len(Poldeg),len(Nsim)))
StdDelta = np.zeros((len(Poldeg),len(Nsim)))
MeanDeltaReg = np.zeros((len(Poldeg),len(Nsim)))
StdDeltaReg = np.zeros((len(Poldeg),len(Nsim)))
MeanPriceForward = np.zeros((len(Poldeg),len(Nsim)))
StdPriceForward = np.zeros((len(Poldeg),len(Nsim)))
MeanPriceRegForward = np.zeros((len(Poldeg),len(Nsim)))
StdPriceRegForward = np.zeros((len(Poldeg),len(Nsim)))
MeanDeltaForward = np.zeros((len(Poldeg),len(Nsim)))
StdDeltaForward = np.zeros((len(Poldeg),len(Nsim)))
MeanDeltaRegForward = np.zeros((len(Poldeg),len(Nsim)))
StdDeltaRegForward = np.zeros((len(Poldeg),len(Nsim)))

for s in range(len(Nsim)):
    n = Nsim[s]
    print("NoSimulations =", n)
    PriceBatch = np.zeros((len(Poldeg), Nbatch))
    PriceBatchReg = np.zeros((len(Poldeg), Nbatch))
    DeltaBatch = np.zeros((len(Poldeg), Nbatch))
    DeltaBatchReg = np.zeros((len(Poldeg), Nbatch))
    PriceBatchForward  = np.zeros((len(Poldeg), NbatchForward))
    PriceBatchRegForward  = np.zeros((len(Poldeg), NbatchForward))
    DeltaBatchForward  = np.zeros((len(Poldeg), NbatchForward))
    DeltaBatchRegForward  = np.zeros((len(Poldeg), NbatchForward))
    for deg in range(len(Poldeg)):
        p1 = Poldeg[deg]
        print("Degree fit = ", p1)
        coeffsAddition = 0
        coeffsAdditionReg = 0
        for k in range(Nbatch):
            M = AmrLSM(spot = S0, capT = 1, Nhedge = 52, NoSim = n, p = p1, NoSeed = k
                       , method = "Delta-regularization")
            PriceBatchReg[deg, k] = M["PriceBackward"]
            DeltaBatchReg[deg, k] = M["DeltaBackward"]
            coeffsAdditionReg = coeffsAdditionReg + M["Coefficients"]
            M = AmrLSM(spot = S0, capT = 1, Nhedge = 52, NoSim = n, p = p1, NoSeed = k
                       , method = "Price-only-regression")
            PriceBatch[deg, k] = M["PriceBackward"]
            DeltaBatch[deg, k] = M["DeltaBackward"]
            coeffsAddition = coeffsAddition + M["Coefficients"]
        coeffsSim = coeffsAddition/Nbatch
        coeffsSimReg = coeffsAdditionReg/Nbatch
        for k in range(NbatchForward):
            M = AmrLSMforward(spot = S0, capT = 1, Nhedge = 52, NoSim = n, p = p1
                              , NoSeed = k, coefficients = coeffsSimReg)
            PriceBatchRegForward[deg, k] = M["PriceForward"]
            DeltaBatchRegForward[deg, k] = M["DeltaForward"]
            M = AmrLSMforward(spot = S0, capT = 1, Nhedge = 52, NoSim = n, p = p1
                              , NoSeed = k, coefficients = coeffsSim)
            PriceBatchForward[deg, k] = M["PriceForward"]
            DeltaBatchForward[deg, k] = M["DeltaForward"]
    for deg in range(len(Poldeg)):
        MeanPrice[deg,s] = np.mean(PriceBatch[deg, :])
        StdPrice[deg,s] = np.std(PriceBatch[deg, :])
        MeanPriceReg[deg,s] = np.mean(PriceBatchReg[deg, :])
        StdPriceReg[deg,s] = np.std(PriceBatchReg[deg, :])
        MeanDelta[deg,s] = np.mean(DeltaBatch[deg, :])
        StdDelta[deg,s] = np.std(DeltaBatch[deg, :])
        MeanDeltaReg[deg,s] = np.mean(DeltaBatchReg[deg, :])
        StdDeltaReg[deg,s] = np.std(DeltaBatchReg[deg, :])

        MeanPriceForward[deg,s] = np.mean(PriceBatchForward[deg, :])
        StdPriceForward[deg,s] = np.std(PriceBatchForward[deg, :])
        MeanPriceRegForward[deg,s] = np.mean(PriceBatchRegForward[deg, :])
        StdPriceRegForward[deg,s] = np.std(PriceBatchRegForward[deg, :])
        MeanDeltaForward[deg,s] = np.mean(DeltaBatchForward[deg, :])
        StdDeltaForward[deg,s] = np.std(DeltaBatchForward[deg, :])
        MeanDeltaRegForward[deg,s] = np.mean(DeltaBatchRegForward[deg, :])
        StdDeltaRegForward[deg,s] = np.std(DeltaBatchRegForward[deg, :])
        
L = AmrPutBino(spot=S0, TimeToMat=1, NoTimePoints=2500)
BinPrice = L["Price"]
BinDelta = L["Delta"]
plt.subplot(1, 2, 1)
plt.plot(Nsim, MeanPrice[0,:]
         , label = "Estimated Backward Price by Price-only-regression", color = "red")
plt.text(Nsim[-1], MeanPrice[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.text(Nsim[0]-250, MeanPrice[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.plot(Nsim, MeanPriceReg[0,:]
         , label = "Estimated Backward Price by Delta-regularization", color = "blue")
plt.text(Nsim[-1], MeanPriceReg[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'blue')
plt.text(Nsim[0]-250, MeanPriceReg[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'blue')

plt.plot(Nsim, MeanPriceForward[0,:]
         , label = "Estimated Forward Price by Price-only-regression", color = "orange")
plt.text(Nsim[-1], MeanPriceForward[0,:][-1]
         , str(Poldeg[0]), fontsize = 10, color = 'orange')
plt.text(Nsim[0]-250, MeanPriceForward[0,:][0]
         , str(Poldeg[0]), fontsize = 10, color = 'orange')
plt.plot(Nsim, MeanPriceRegForward[0,:]
         , label = "Estimated Forward Price by Delta-regularization", color = "green")
plt.text(Nsim[-1], MeanPriceRegForward[0,:][-1]
         , str(Poldeg[0]), fontsize = 10, color = 'green')
plt.text(Nsim[0]-250, MeanPriceRegForward[0,:][0]
         , str(Poldeg[0]), fontsize = 10, color = 'green')
for i in range(1,len(Poldeg)):
    plt.plot(Nsim, MeanPrice[i,:], color = "red")
    plt.text(Nsim[-1], MeanPrice[i,:][-1]
             , str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.text(Nsim[0]-250, MeanPrice[i,:][0]
             , str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.plot(Nsim, MeanPriceReg[i,:], color = "blue")
    plt.text(Nsim[-1], MeanPriceReg[i,:][-1]
             , str(Poldeg[i]), fontsize = 10, color = 'blue')
    plt.text(Nsim[0]-250, MeanPriceReg[i,:][0]
             , str(Poldeg[i]), fontsize = 10, color = 'blue')
    
    plt.plot(Nsim, MeanPriceForward[i,:], color = "orange")
    plt.text(Nsim[-1], MeanPriceForward[i,:][-1]
             , str(Poldeg[i]), fontsize = 10, color = 'orange')
    plt.text(Nsim[0]-250, MeanPriceForward[i,:][0]
             , str(Poldeg[i]), fontsize = 10, color = 'orange')
    plt.plot(Nsim, MeanPriceRegForward[i,:], color = "green")
    plt.text(Nsim[-1], MeanPriceRegForward[i,:][-1]
             , str(Poldeg[i]), fontsize = 10, color = 'green')
    plt.text(Nsim[0]-250, MeanPriceRegForward[i,:][0]
             , str(Poldeg[i]), fontsize = 10, color = 'green')
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

plt.plot(Nsim, StdPriceForward[0,:]
         , label = "Estimated standard error", color = "orange")
plt.text(Nsim[-1], StdPriceForward[0,:][-1]
         , str(Poldeg[0]), fontsize = 10, color = 'orange')
plt.text(Nsim[0]-50, StdPriceForward[0,:][0]
         , str(Poldeg[0]), fontsize = 10, color = 'orange')
plt.plot(Nsim, StdPriceRegForward[0,:]
         , label = "Estimated standard error", color = "green")
plt.text(Nsim[-1], StdPriceRegForward[0,:][-1]
         , str(Poldeg[0]), fontsize = 10, color = 'green')
plt.text(Nsim[0]-50, StdPriceRegForward[0,:][0]
         , str(Poldeg[0]), fontsize = 10, color = 'green')
for i in range(1,len(Poldeg)):
    plt.plot(Nsim, StdPrice[i,:], color = "red")
    plt.text(Nsim[-1], StdPrice[i,:][-1], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.text(Nsim[0]-50, StdPrice[i,:][0], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.plot(Nsim, StdPriceReg[i,:], color = "blue")
    plt.text(Nsim[-1], StdPriceReg[i,:][-1]
             , str(Poldeg[i]), fontsize = 10, color = 'blue')
    plt.text(Nsim[0]-50, StdPriceReg[i,:][0]
             , str(Poldeg[i]), fontsize = 10, color = 'blue')

    plt.plot(Nsim, StdPriceForward[i,:], color = "orange")
    plt.text(Nsim[-1], StdPriceForward[i,:][-1]
             , str(Poldeg[i]), fontsize = 10, color = 'orange')
    plt.text(Nsim[0]-50, StdPriceForward[i,:][0]
             , str(Poldeg[i]), fontsize = 10, color = 'orange')
    plt.plot(Nsim, StdPriceRegForward[i,:], color = "green")
    plt.text(Nsim[-1], StdPriceRegForward[i,:][-1]
             , str(Poldeg[i]), fontsize = 10, color = 'green')
    plt.text(Nsim[0]-50, StdPriceRegForward[i,:][0]
             , str(Poldeg[i]), fontsize = 10, color = 'green')
    
plt.axhline(y=0, color='black', linestyle='-')
plt.xlabel('NoSimulations')
plt.ylabel('Standard error')
plt.legend()
plt.show()
plt.subplot(1, 2, 1)
plt.plot(Nsim, MeanDelta[0,:]
         , label = "Estimated Backward Delta by Price-only-regression", color = "red")
plt.text(Nsim[-1], MeanDelta[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.text(Nsim[0]-250, MeanDelta[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.plot(Nsim, MeanDeltaReg[0,:]
         , label = "Estimated Backward Delta by Delta-regularization", color = "blue")
plt.text(Nsim[-1], MeanDeltaReg[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'blue')
plt.text(Nsim[0]-250, MeanDeltaReg[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'blue')

plt.plot(Nsim, MeanDeltaForward[0,:]
         , label = "Estimated Forward Delta by Price-only-regression", color = "orange")
plt.text(Nsim[-1], MeanDeltaForward[0,:][-1]
         , str(Poldeg[0]), fontsize = 10, color = 'orange')
plt.text(Nsim[0]-250, MeanDeltaForward[0,:][0]
         , str(Poldeg[0]), fontsize = 10, color = 'orange')
plt.plot(Nsim, MeanDeltaRegForward[0,:]
         , label = "Estimated Forward Delta by Delta-regularization", color = "green")
plt.text(Nsim[-1], MeanDeltaRegForward[0,:][-1]
         , str(Poldeg[0]), fontsize = 10, color = 'green')
plt.text(Nsim[0]-250, MeanDeltaRegForward[0,:][0]
         , str(Poldeg[0]), fontsize = 10, color = 'green')
for i in range(1,len(Poldeg)):
    plt.plot(Nsim, MeanDelta[i,:], color = "red")
    plt.text(Nsim[-1], MeanDelta[i,:][-1]
             , str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.text(Nsim[0]-250, MeanDelta[i,:][0]
             , str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.plot(Nsim, MeanDeltaReg[i,:], color = "blue")
    plt.text(Nsim[-1], MeanDeltaReg[i,:][-1]
             , str(Poldeg[i]), fontsize = 10, color = 'blue')
    plt.text(Nsim[0]-250, MeanDeltaReg[i,:][0]
             , str(Poldeg[i]), fontsize = 10, color = 'blue')

    plt.plot(Nsim, MeanDeltaForward[i,:], color = "orange")
    plt.text(Nsim[-1], MeanDeltaForward[i,:][-1]
             , str(Poldeg[i]), fontsize = 10, color = 'orange')
    plt.text(Nsim[0]-250, MeanDeltaForward[i,:][0]
             , str(Poldeg[i]), fontsize = 10, color = 'orange')
    plt.plot(Nsim, MeanDeltaRegForward[i,:], color = "green")
    plt.text(Nsim[-1], MeanDeltaRegForward[i,:][-1]
             , str(Poldeg[i]), fontsize = 10, color = 'green')
    plt.text(Nsim[0]-250, MeanDeltaRegForward[i,:][0]
             , str(Poldeg[i]), fontsize = 10, color = 'green')
plt.axhline(y=BinDelta, color='black', linestyle='-', label = "Binomial Delta")
plt.xlabel('NoSimulations')
plt.ylabel('Time-zero Delta')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(Nsim, StdDelta[0,:], label = "Estimated standard error", color = "red")
plt.text(Nsim[-1], StdDelta[0,:][-1], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.text(Nsim[0]-50, StdDelta[0,:][0], str(Poldeg[0]), fontsize = 10, color = 'red')
plt.plot(Nsim, StdDeltaReg[0,:], label = "Estimated standard error", color = "blue")
plt.text(Nsim[-1], StdDeltaReg[0,:][-1]
         , str(Poldeg[0]), fontsize = 10, color = 'blue')
plt.text(Nsim[0]-50, StdDeltaReg[0,:][0]
         , str(Poldeg[0]), fontsize = 10, color = 'blue')

plt.plot(Nsim, StdDeltaForward[0,:]
         , label = "Estimated standard error", color = "orange")
plt.text(Nsim[-1], StdDeltaForward[0,:][-1]
         , str(Poldeg[0]), fontsize = 10, color = 'orange')
plt.text(Nsim[0]-50, StdDeltaForward[0,:][0]
         , str(Poldeg[0]), fontsize = 10, color = 'orange')
plt.plot(Nsim, StdDeltaRegForward[0,:]
         , label = "Estimated standard error", color = "green")
plt.text(Nsim[-1], StdDeltaRegForward[0,:][-1]
         , str(Poldeg[0]), fontsize = 10, color = 'green')
plt.text(Nsim[0]-50, StdDeltaRegForward[0,:][0]
         , str(Poldeg[0]), fontsize = 10, color = 'green')
for i in range(1,len(Poldeg)):
    plt.plot(Nsim, StdDelta[i,:], color = "red")
    plt.text(Nsim[-1], StdDelta[i,:][-1], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.text(Nsim[0]-50, StdDelta[i,:][0], str(Poldeg[i]), fontsize = 10, color = 'red')
    plt.plot(Nsim, StdDeltaReg[i,:], color = "blue")
    plt.text(Nsim[-1], StdDeltaReg[i,:][-1]
             , str(Poldeg[i]), fontsize = 10, color = 'blue')
    plt.text(Nsim[0]-50, StdDeltaReg[i,:][0]
             , str(Poldeg[i]), fontsize = 10, color = 'blue')

    plt.plot(Nsim, StdDeltaForward[i,:], color = "orange")
    plt.text(Nsim[-1], StdDeltaForward[i,:][-1]
             , str(Poldeg[i]), fontsize = 10, color = 'orange')
    plt.text(Nsim[0]-50, StdDeltaForward[i,:][0]
             , str(Poldeg[i]), fontsize = 10, color = 'orange')
    plt.plot(Nsim, StdDeltaRegForward[i,:], color = "green")
    plt.text(Nsim[-1], StdDeltaRegForward[i,:][-1]
             , str(Poldeg[i]), fontsize = 10, color = 'green')
    plt.text(Nsim[0]-50, StdDeltaRegForward[i,:][0]
             , str(Poldeg[i]), fontsize = 10, color = 'green')
plt.axhline(y=0, color='black', linestyle='-')
plt.xlabel('NoSimulations')
plt.ylabel('Standard error')
plt.legend()
plt.show()
def PriceForwardConfInterval(degree):
    upperboundRegForward = (MeanPriceRegForward
                            - sc.norm.ppf(0.975)*StdPriceRegForward)
    lowerboundRegForward = (MeanPriceRegForward
                            + sc.norm.ppf(0.975)*StdPriceRegForward)
    upperboundDegForward = (MeanPriceForward - sc.norm.ppf(0.975)*StdPriceForward)
    lowerboundDegForward = (MeanPriceForward + sc.norm.ppf(0.975)*StdPriceForward)
    plt.plot()
    plt.plot(Nsim, MeanPriceForward[degree,:]
             , label = "Estimated Forward price by price-only-regression", color = "orange")
    plt.text(Nsim[-1], MeanPriceForward[degree,:][-1]
             , str(Poldeg[degree]), fontsize = 10, color = 'orange')
    plt.plot(Nsim, upperboundDegForward[degree,:]
             , label = "95% confidence interval", color = "orange", linestyle = "dashed")
    plt.plot(Nsim, lowerboundDegForward[degree,:],color = "orange",  linestyle = "dashed")
    plt.plot(Nsim, MeanPriceRegForward[degree,:]
             , label = "Estimated Forward price by Delta-regularization", color = "green")
    plt.text(Nsim[-1], MeanPriceRegForward[degree,:][-1]
             , str(Poldeg[degree]), fontsize = 10, color = 'green')
    plt.plot(Nsim, upperboundRegForward[degree,:]
             , label = "95% confidence interval", color = "green", linestyle = "dashed")
    plt.plot(Nsim, lowerboundRegForward[degree,:],color = "green",  linestyle = "dashed")
    plt.axhline(y=BinPrice, color='black', linestyle='-', label = "Binomial Price")
    plt.xlabel('NoSimulations')
    plt.ylabel('Time-zero Price')
    plt.title("American put-Price by " + str(Poldeg[degree])
              +"-degree polynomial fit given S(0)=" + str(S0))
    plt.legend()
    plt.show()
def DeltaForwardConfInterval(degree):
    upperboundRegForward = (MeanDeltaRegForward
                            - sc.norm.ppf(0.975)*StdDeltaRegForward)
    lowerboundRegForward = (MeanDeltaRegForward
                            + sc.norm.ppf(0.975)*StdDeltaRegForward)
    upperboundDegForward = MeanDeltaForward - sc.norm.ppf(0.975)*StdDeltaForward
    lowerboundDegForward = MeanDeltaForward + sc.norm.ppf(0.975)*StdDeltaForward
    plt.plot()
    
    plt.plot(Nsim, MeanDeltaForward[degree,:]
             , label = "Estimated Forward Delta by Price-only-regression", color = "orange")
    plt.text(Nsim[-1], MeanDeltaForward[degree,:][-1]
             , str(Poldeg[degree]), fontsize = 10, color = 'orange')
    plt.plot(Nsim, upperboundDegForward[degree,:], label = "95% confidence interval"
             , color = "orange", linestyle = "dashed")
    plt.plot(Nsim, lowerboundDegForward[degree,:],color = "orange",  linestyle = "dashed")
    plt.plot(Nsim, MeanDeltaRegForward[degree,:]
             , label = "Estimated Forward Delta by Delta-regularization", color = "green")
    plt.text(Nsim[-1], MeanDeltaRegForward[degree,:][-1], str(Poldeg[degree]), fontsize = 10
             , color = 'green')
    plt.plot(Nsim, upperboundRegForward[degree,:], label = "95% confidence interval"
             , color = "green", linestyle = "dashed")
    plt.plot(Nsim, lowerboundRegForward[degree,:],color = "green",  linestyle = "dashed")
    plt.axhline(y=BinDelta, color='black', linestyle='-', label = "Binomial Delta")
    plt.xlabel('NoSimulations')
    plt.ylabel('Time-zero Delta')
    plt.title("American put-Delta by " + str(Poldeg[degree])
              +"-degree polynomial fit given S(0)=" + str(S0))
    plt.legend()
    plt.show()
