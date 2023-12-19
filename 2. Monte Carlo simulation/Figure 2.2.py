# This code is similar to the code producing Figure 2.1
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
N = 1000
S0 = 1
# Black-Sclosed-form formula for calculation the Delta of a
# European put option
def TrueDelta(price, TimeToMat):
    d1 = (np.log(price/K) + (r + 0.5*sigma**2)*TimeToMat)/(sigma*np.sqrt(TimeToMat))
    result = -sc.norm.cdf(-d1)
    return(result)
DeltasSim = np.zeros((N,len(Nsim)))
for j in range(N):
    Deltas = np.zeros(len(Nsim))
    for s in range(len(Nsim)):
        n = Nsim[s]
        Z = np.random.normal(0, 1, size=n)
        ST = S0*np.exp((r-0.5*sigma**2)*capT + sigma*np.sqrt(capT)*Z)
        D0 = -1*np.exp(-r*capT)*(ST<K)*ST/S0
        Delta0 = np.mean(D0)
        Deltas[s] = Delta0
    DeltasSim[j,:] = Deltas
Deltas = np.zeros(len(Nsim))
StdDeltas = np.zeros(len(Nsim))
for s in range(len(Nsim)):
    n = Nsim[s]
    DeltasBatch = np.zeros(Nbatch)
    for k in range(Nbatch):
        Z = np.random.normal(0, 1, size=n)
        ST = S0*np.exp((r-0.5*sigma**2)*capT + sigma*np.sqrt(capT)*Z)
        # Simulation the random variable exact that gives European put Delta in expectation
        D0 = -1*np.exp(-r*capT)*(ST<K)*ST/S0
        Delta0 = np.mean(D0)
        DeltasBatch[k] = Delta0
    Deltas[s] = np.mean(DeltasBatch)
    StdDeltas[s] = np.std(DeltasBatch)
lowerbound = (TrueDelta(S0,capT) - sc.norm.ppf(0.975)*StdDeltas)
upperbound = (TrueDelta(S0,capT) + sc.norm.ppf(0.975)*StdDeltas)
numOverUpperBound = np.zeros(len(Nsim))
numUnderLowerUpperBound = np.zeros(len(Nsim))
for s in range(len(Nsim)):
    NoOverUpperBound = (1*(DeltasSim[:,s]>upperbound[s])).sum()
    numOverUpperBound[s] = NoOverUpperBound/N
    NoUnderLowerBound = (1*(DeltasSim[:,s]<lowerbound[s])).sum()
    numUnderLowerUpperBound[s] = NoUnderLowerBound/N
NumBreachesConfidenceInterval = numOverUpperBound + numUnderLowerUpperBound
for j in range(N):
    plt.scatter(Nsim, DeltasSim[j,:], color = "green")
plt.plot(Nsim, Deltas, color = "blue", label = "Approximating the Delta")
plt.plot(Nsim, upperbound, label = "Approx 95% confidence interval"
         , color = "blue", linestyle = "dashed")
plt.plot(Nsim, lowerbound, color = "blue", linestyle = "dashed")
for s in range(len(Nsim)):
    plt.text(Nsim[s],DeltasSim[:,s].max()
             ,str(np.round(NumBreachesConfidenceInterval[s],4)), fontsize = 10)
plt.text(Nsim[2],DeltasSim[:,1].min(), "Number of Batching = " + str(Nbatch)
         , fontsize = 10)
plt.axhline(y=TrueDelta(S0,capT), color='black', linestyle='-', label = "True Delta")
plt.xlabel('Number of simulations')
plt.ylabel('Time-zero Delta')
plt.legend()
plt.show()

