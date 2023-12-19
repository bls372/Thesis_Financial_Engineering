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
def EUPrice(price, TimeToMat):
    d1 = (np.log(price/K) + (r + 0.5*sigma**2)*TimeToMat)/(sigma*np.sqrt(TimeToMat))
    d2 = d1 - sigma*np.sqrt(TimeToMat)
    result = np.exp(-r*TimeToMat)*K*sc.norm.cdf(-d2) - price*sc.norm.cdf(-d1)
    return(result)
def EUDelta(price, TimeToMat):
    d1 = (np.log(price/K) + (r + 0.5*sigma**2)*TimeToMat)/(sigma*np.sqrt(TimeToMat))
    result = -sc.norm.cdf(-d1)
    return(result)
NoTimeSteps = np.linspace(start=5, stop=1000, num=50)
N = len(NoTimeSteps)
BinPrices = np.zeros(N)
BinDeltas = np.zeros(N)
M = AmrPutBino(spot=S0, TimeToMat=capT, NoTimePoints=2500)
BinPrice = M["Price"]
BinDelta = M["Delta"]
for i in range(N):
    M = AmrPutBino(spot=S0, TimeToMat=capT, NoTimePoints=int(NoTimeSteps[i]))
    BinPrices[i] = M["Price"]
    BinDeltas[i] = M["Delta"]
plt.scatter(NoTimeSteps, BinPrices, label = "Binomial price", color = "red")
plt.axhline(y=BinPrice, color='black', linestyle='-'
            , label = "Binomial Price with 2500 time-steps")
plt.xlabel('Number of time-steps')
plt.ylabel('Time-zero Price')
plt.title('American Put option with K = ' + str(K) + ' and S0 =' + str(S0))
plt.legend()
plt.show()
plt.scatter(NoTimeSteps, BinDeltas, label = "Binomial Delta", color = "blue")
plt.axhline(y=BinDelta, color='black', linestyle='-'
            , label = "Binomial Delta with 2500 time-steps")
plt.xlabel('Number of time-steps')
plt.ylabel('Time-zero Delta')
plt.title('American Put option with K = ' + str(K) + ' and S0 =' + str(S0))
plt.legend()
plt.show()
TimeStep = 250
N=1000
Z = np.random.normal(0, 1, size=N)
S0s = K*np.exp((r-0.5*sigma**2)*capT + sigma*np.sqrt(capT)*Z)
BinPrices = np.zeros(N)
BinDeltas = np.zeros(N)
for i in range(N):
    M = AmrPutBino(spot=S0s[i], TimeToMat=capT, NoTimePoints=TimeStep)
    BinPrices[i] = M["Price"]
    BinDeltas[i] = M["Delta"]
BSprices = EUPrice(S0s,capT)
BSdeltas = EUDelta(S0s,capT)
data = {"S0": S0s, "BinPrice": BinPrices, "TruePrice": BSprices
        , "BinDelta": BinDeltas, "TrueDelta": BSdeltas}
df = pd.DataFrame(data)
dfsort = df.sort_values("S0")
plt.plot(dfsort["S0"], dfsort["TruePrice"]
         , label = "Black-Scholes European Price", color ="black")
plt.plot(dfsort["S0"], dfsort["BinPrice"]
         , label = "Binomial American price with " + str(TimeStep)
         + " time-steps", color = "red")
plt.xlabel('Time-zero Stock-price')
plt.ylabel('Time-zero Amr-put Price')
plt.title('American Put option with K = ' + str(K))
plt.legend()
plt.show()
plt.plot(dfsort["S0"], dfsort["TrueDelta"]
         , label = "Black-Scholes Delta", color ="black")
plt.plot(dfsort["S0"], dfsort["BinDelta"]
         , label =  "Binomial American Delta with " + str(TimeStep)
         + " time-steps", color = "blue")
plt.xlabel('Time-zero Stock-price')
plt.ylabel('Time-zero Amr-put Delta')
plt.title('American Put option with K = ' + str(K))
plt.legend()
plt.show()


