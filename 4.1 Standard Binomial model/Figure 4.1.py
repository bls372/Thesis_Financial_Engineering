import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sc
S0=1
K = 1
sigma = 0.2
r = 0.06
capT = 1
Nhedge = 52
dt = capT/Nhedge
def TruePrice(price, TimeToMat):
    d1 = (np.log(price/K) + (r + 0.5*sigma**2)*TimeToMat)/(sigma*np.sqrt(TimeToMat))
    d2 = d1 - sigma*np.sqrt(TimeToMat)
    result = np.exp(-r*TimeToMat)*K*sc.norm.cdf(-d2) - price*sc.norm.cdf(-d1)
    return(result)
def TrueDelta(price, TimeToMat):
    d1 = (np.log(price/K) + (r + 0.5*sigma**2)*TimeToMat)/(sigma*np.sqrt(TimeToMat))
    result = -sc.norm.cdf(-d1)
    return(result)
def EuPutBino(spot, TimeToMat, NoTimePoints):
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
            cashflow[j][l-1] = (1/R)*(q*cashflow[j][l] + (1-q)*cashflow[j+1][l])

    Delta = (cashflow[0][1]-cashflow[1][1])/(St[0][1]-St[1][1])
    result = {"Price": cashflow[0][0], "Delta": Delta}
    return(result)
NoTimeSteps = np.linspace(start=5, stop=1000, num=50)
N = len(NoTimeSteps)
BinPrices = np.zeros(N)
BinDeltas = np.zeros(N)
for i in range(N):
    M = EuPutBino(spot=S0, TimeToMat=capT, NoTimePoints=int(NoTimeSteps[i]))
    BinPrices[i] = M["Price"]
    BinDeltas[i] = M["Delta"]
plt.scatter(NoTimeSteps, BinPrices, label = "Binomial price", color = "red")
plt.axhline(y=TruePrice(S0,capT), color='black', linestyle='-'
            , label = "Black-Scholes Price")
plt.xlabel('Number of time-steps')
plt.ylabel('Time-zero Price')
plt.title('European Put option with K = ' + str(K) + ' and S0 =' + str(S0))
plt.legend()
plt.show()
plt.scatter(NoTimeSteps, BinDeltas, label = "Binomial Delta", color = "blue")
plt.axhline(y=TrueDelta(S0,capT), color='black', linestyle='-'
            , label = "Black-Scholes Delta")
plt.xlabel('Number of time-steps')
plt.ylabel('Time-zero Delta')
plt.title('European Put option with K = ' + str(K) + ' and S0 =' + str(S0))
plt.legend()
plt.show()
N=1000
TimeSteps = 50
Z = np.random.normal(0, 1, size=N)
S0s = K*np.exp((r-0.5*sigma**2)*capT + sigma*np.sqrt(capT)*Z)
BinPrices = np.zeros(N)
BinDeltas = np.zeros(N)
for i in range(N):
    M = EuPutBino(spot=S0s[i], TimeToMat=capT, NoTimePoints=TimeSteps)
    BinPrices[i] = M["Price"]
    BinDeltas[i] = M["Delta"]
BSprices = TruePrice(S0s,capT)
BSdeltas = TrueDelta(S0s,capT)
data = {"S0": S0s, "BinPrice": BinPrices, "TruePrice": BSprices
        , "BinDelta": BinDeltas, "TrueDelta": BSdeltas}
df = pd.DataFrame(data)
dfsort = df.sort_values("S0")
plt.plot(dfsort["S0"], dfsort["TruePrice"], label = "Black-Scholes Price"
         , color ="black")
plt.plot(dfsort["S0"], dfsort["BinPrice"]
         , label = "Binomial price with " + str(TimeSteps) + " time-steps"
         , color = "red")
plt.xlabel('Time-zero Stock-price')
plt.ylabel('Time-zero EU-put Price')
plt.title('European Put option with K = ' + str(K))
plt.legend()
plt.show()
plt.plot(dfsort["S0"], dfsort["TrueDelta"], label = "Black-Scholes Price"
         , color ="black")
plt.plot(dfsort["S0"], dfsort["BinDelta"]
         , label = "Binomial Delta with " + str(TimeSteps) + " time-steps"
         , color = "blue")
plt.xlabel('Time-zero Stock-price')
plt.ylabel('Time-zero EU-put Delta')
plt.title('European Put option with K = ' + str(K))
plt.legend()
plt.show()
