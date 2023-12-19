import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sc  
K = 1
sigma = 0.2
r = 0.06
w = 0.5
n = 2000
p = 9
Nhedge = 52
capT = 1
dt = capT/Nhedge
ZeroPriceTime = 0
def TruePrice(price, TimeToMat):
    d1 = (np.log(price/K) + (r + 0.5*sigma**2)*TimeToMat)/(sigma*np.sqrt(TimeToMat))
    d2 = d1 - sigma*np.sqrt(TimeToMat)
    result = np.exp(-r*TimeToMat)*K*sc.norm.cdf(-d2) - price*sc.norm.cdf(-d1)
    return(result)
def TrueDelta(price, TimeToMat):
    d1 = (np.log(price/K) + (r + 0.5*sigma**2)*TimeToMat)/(sigma*np.sqrt(TimeToMat))
    result = -sc.norm.cdf(-d1)
    return(result)
coeffsDeg = np.zeros((Nhedge,  p+1))
coeffsReg = np.zeros((Nhedge, p+1))
Ds = np.zeros((n, Nhedge))
DiscPayoffs = np.zeros((n, Nhedge))
Z = np.random.normal(0, 1, size=n)
S0 = K*np.exp((r-0.5*sigma**2)*capT + sigma*np.sqrt(capT)*Z)
St = np.zeros((n, Nhedge +1))
St[:,0] = S0      
for i in range(Nhedge):
    Z = np.random.normal(0, 1, size=n)
    St[:,i+1] = St[:,i]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
ST = St[:, Nhedge]
payoff = (K-ST)*(ST<K)
for i in range(Nhedge):
    TimeToMat = (capT-i*dt)
    S0 = St[:, i]
    X = np.vander((S0-K), p+1, increasing=True)
    Y = np.c_[np.zeros(n), np.ones(n), range(2,p+1)*np.delete(X, [0,p], 1)]
    P = payoff*np.exp(-r*TimeToMat)
    DiscPayoffs[:,i] = P
    coeffsDeg[i,:] = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X))
                               , np.matmul(np.transpose(X), P))
    D = -1*np.exp(-r*TimeToMat)*(ST<K)*ST/S0
    Ds[:,i] = D
    coeffsReg[i,:]= np.matmul(np.linalg.inv(w*np.matmul(np.transpose(X),X)
                                            + (1-w)*np.matmul(np.transpose(Y), Y))
                              , w*np.matmul(np.transpose(X), P)
                              + (1-w)*np.matmul(np.transpose(Y), D))
Z = np.random.normal(0, 1, size=1000)
S01 = K*np.exp((r-0.5*sigma**2)*(0.7) + sigma*np.sqrt(0.7)*Z)
St1 = np.zeros((1000, Nhedge))
St1[:,0] = S01
for i in range(Nhedge-1):
    Z = np.random.normal(0, 1, size=1000)
    St1[:,i+1] = St1[:,i]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
S = St1[:,ZeroPriceTime]
X = np.vander(S-K, p+1, increasing=True)
Xdiff = np.c_[np.zeros(len(S)), np.ones(len(S)), range(2,p+1)*np.delete(X, [0,p], 1)]
fitPriceDeg = np.matmul(X, coeffsDeg[ZeroPriceTime, :])
fitDeltaDeg = np.matmul(Xdiff, coeffsDeg[ZeroPriceTime, :])
fitPriceReg = np.matmul(X, coeffsReg[ZeroPriceTime, :])
fitDeltaReg = np.matmul(Xdiff, coeffsReg[ZeroPriceTime, :])
prices = TruePrice(S, capT-ZeroPriceTime/Nhedge)
deltas = TrueDelta(S, capT-ZeroPriceTime/Nhedge)
data = {"S0": S, "PricefitDeg": fitPriceDeg, "PricefitReg": fitPriceReg
        , "DeltafitDeg": fitDeltaDeg, "DeltafitReg": fitDeltaReg
        , "TruePrice": prices, "TrueDelta": deltas}
df = pd.DataFrame(data)
dfsort = df.sort_values("S0")
plt.subplot(1, 2, 1)
plt.scatter(St[:, ZeroPriceTime], DiscPayoffs[:,ZeroPriceTime], color = "grey"
            , label = "Simulated  Put-payoffs")
plt.plot(dfsort["S0"], dfsort["PricefitDeg"], label = "Price-only-regression"
         , color = "red")
plt.plot(dfsort["S0"], dfsort["PricefitReg"], label = "Delta-regularization"
         , color = "blue")
plt.plot(dfsort["S0"], dfsort["TruePrice"], label = "TruePrice", color = "black")
plt.xlabel('Time-zero-prices')
plt.ylabel('Put-payoff')
plt.title('Fitting Price')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(St[:, ZeroPriceTime], Ds[:,ZeroPriceTime], color = "grey"
            , label = "Simulated Deltas")
plt.plot(dfsort["S0"], dfsort["DeltafitDeg"], label = "Price-only-regression"
         , color = "red")
plt.plot(dfsort["S0"], dfsort["DeltafitReg"], label = "Delta-regularization"
         , color = "blue")
plt.plot(dfsort["S0"], dfsort["TrueDelta"], label = "TrueDelta", color = "black")
plt.xlabel('Time-zero-prices')
plt.ylabel('Delta')                      
plt.title('Fitting Delta')
plt.legend()
plt.show()

