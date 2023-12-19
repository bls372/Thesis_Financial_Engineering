import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sc  
K = 1
sigma = 0.2
r = 0.06
capT = 1
Nsim = np.array([500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
Nbatch = 250
# N is the number of new approximations of the price
N = 1000
S0 = 1
#Black-Scholes closed form formula for European Put options
def TruePrice(price, TimeToMat):
    d1 = (np.log(price/K) + (r + 0.5*sigma**2)*TimeToMat)/(sigma*np.sqrt(TimeToMat))
    d2 = d1 - sigma*np.sqrt(TimeToMat)
    result = np.exp(-r*TimeToMat)*K*sc.norm.cdf(-d2) - price*sc.norm.cdf(-d1)
    return(result)
#N number of new approximations of the price for each number of simulations
pricesSim = np.zeros((N,len(Nsim)))
for j in range(N):
    prices = np.zeros(len(Nsim))
    for s in range(len(Nsim)):
        n = Nsim[s]
        Z = np.random.normal(0, 1, size=n)
        ST = S0*np.exp((r-0.5*sigma**2)*capT + sigma*np.sqrt(capT)*Z)
        Discpayoffs = np.exp(-r*capT)*(K-ST)*(ST<K)
        Price0 = np.mean(Discpayoffs)
        prices[s] = Price0
    pricesSim[j,:] = prices
#Mean-prices from the Batch experiment for each number of simulations
prices = np.zeros(len(Nsim))
#Standard errors of the Approximated price from the Batch experiment
Stdprices = np.zeros(len(Nsim))
for s in range(len(Nsim)):
    n = Nsim[s]
    pricesBatch = np.zeros(Nbatch)
    for k in range(Nbatch):
        Z = np.random.normal(0, 1, size=n)
        ST = S0*np.exp((r-0.5*sigma**2)*capT + sigma*np.sqrt(capT)*Z)
        Discpayoffs = np.exp(-r*capT)*(K-ST)*(ST<K)
        Price0 = np.mean(Discpayoffs)
        pricesBatch[k] = Price0
    prices[s] = np.mean(pricesBatch)
    Stdprices[s] = np.std(pricesBatch)
#Calculating the lower bound in 95% Normal confidence interval
lowerbound = (TruePrice(S0,capT) - sc.norm.ppf(0.975)*Stdprices)
#Calculating the upper bound in 95% Normal confidence interval
upperbound = (TruePrice(S0,capT) + sc.norm.ppf(0.975)*Stdprices)
#Now it is calculated how many of the new price approximations breaches
# the confidence interval for each number of simulations
numOverUpperBound = np.zeros(len(Nsim))
numUnderLowerUpperBound = np.zeros(len(Nsim))
for s in range(len(Nsim)):
    NoOverUpperBound = (1*(pricesSim[:,s]>upperbound[s])).sum()
    numOverUpperBound[s] = NoOverUpperBound/N
    NoUnderLowerBound = (1*(pricesSim[:,s]<lowerbound[s])).sum()
    numUnderLowerUpperBound[s] = NoUnderLowerBound/N
# The total number of breaches is the sum of many there are over the upper bound
# plus how many there are below the lower bound
NumBreachesConfidenceInterval = numOverUpperBound + numUnderLowerUpperBound
for j in range(N):
    plt.scatter(Nsim, pricesSim[j,:], color = "yellow")
plt.plot(Nsim, prices, color = "red", label = "Approximating the price")
plt.plot(Nsim, upperbound, label = "Approx 95% confidence interval"
         , color = "red", linestyle = "dashed")
plt.plot(Nsim, lowerbound, color = "red", linestyle = "dashed")
for s in range(len(Nsim)):
    plt.text(Nsim[s],pricesSim[:,s].max()
             ,str(np.round(NumBreachesConfidenceInterval[s],4)), fontsize = 10)
plt.text(Nsim[2],pricesSim[:,1].min(), "Number of Batching = " + str(Nbatch)
         , fontsize = 10)
plt.axhline(y=TruePrice(S0,capT), color='black', linestyle='-', label = "True Price")
plt.xlabel('Number of simulations')
plt.ylabel('Time-zero price')
plt.legend()
plt.show()
