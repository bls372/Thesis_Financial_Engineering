import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sc
K = 1
r = 0.06
sigma = 0.2
capT = 1
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
def AmrLSM(capT, Nhedge, NoSim, p, NoSeed, method):
    n = NoSim
    dt = capT/Nhedge
    w = 0.5
    np.random.seed(seed=NoSeed+10000)
    St = np.zeros((n, Nhedge +1))
    Z = np.random.normal(0, 1, size=n)
    S0 = K*np.exp((r-0.5*sigma**2)*capT + sigma*np.sqrt(capT)*Z)
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
                                            +(1-w)*np.matmul(np.transpose(Xdiff), Xdiff))
                              , w*np.matmul(np.transpose(X),Y)
                              +(1-w)*np.matmul(np.transpose(Xdiff),D_in))
        contVal = np.matmul(X, coeff)
        exVal = PayOffMatrix[inMonPosition,q]
        cashflowTJEK[inMonPosition] = exVal*(exVal>=contVal)*1 + Y*(exVal<contVal)
        STstop[inMonPosition] = x*(exVal>=contVal)*1+ discST_in*(exVal<contVal)
        ST[inMonPosition] = x*(exVal>=contVal)*1 + ST_in*(exVal<contVal)
        cashflowTJEK = np.exp(-r*dt)*cashflowTJEK
        STstop = np.exp(-r*dt)*STstop
    discST = STstop
    D = -1*(ST<K)*discST/S0
    Y = cashflowTJEK
    X = np.vander(S0-K, p+1, increasing=True)
    Xdiff = np.c_[np.zeros(len(S0)), np.ones(len(S0)), range(2,p+1)
                  *np.delete(X, [0,p], 1)]
    if(method == "Price-only-regression"):
        coeff = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X))
                          , np.matmul(np.transpose(X), Y))
    if(method == "Delta-regularization"):
        coeff = np.matmul(np.linalg.inv(w*np.matmul(np.transpose(X),X)
                                        +(1-w)*np.matmul(np.transpose(Xdiff), Xdiff))
                          , w*np.matmul(np.transpose(X),Y)
                          +(1-w)*np.matmul(np.transpose(Xdiff),D))
    result = {"Coefficients": coeff, "DiscPayoff": Y, "D": D, "S0": S0}
    return(result)
Nsim = 10000
p1 = 10
M = AmrLSM(capT=1, Nhedge=52, NoSim=Nsim, p=p1, NoSeed = 10, method="Price-only-regression")
coeffs = M["Coefficients"]
DiscPayoff = M["DiscPayoff"]
M = AmrLSM(capT=1, Nhedge=52, NoSim=Nsim, p=p1, NoSeed = 10, method="Delta-regularization")
coeffsReg = M["Coefficients"]
DiscPayoffReg = M["DiscPayoff"]
D = M["D"]
S0 = M["S0"]
TimeStep = 50
N=1000
Z = np.random.normal(0, 1, size=N)
S0s = K*np.exp((r-0.5*sigma**2)*capT + sigma*np.sqrt(capT)*Z)
BinPrices = np.zeros(N)
BinDeltas = np.zeros(N)
for i in range(N):
    M = AmrPutBino(spot=S0s[i], TimeToMat=capT, NoTimePoints=TimeStep)
    BinPrices[i] = M["Price"]
    BinDeltas[i] = M["Delta"]

X = np.vander(S0s-K, p1+1, increasing=True)
Xdiff = np.c_[np.zeros(len(S0s)), np.ones(len(S0s)), range(2,p1+1)
                  *np.delete(X, [0,p1], 1)]
FittedPrices = np.matmul(X, coeffs)
FittedDeltas = np.matmul(Xdiff, coeffs)
FittedPricesReg = np.matmul(X, coeffsReg)
FittedDeltasReg = np.matmul(Xdiff, coeffsReg)

data = {"S0": S0s, "BinPrice": BinPrices, "BinDelta": BinDeltas
        , "FittedPrice": FittedPrices, "FittedDelta": FittedDeltas
        , "FittedPriceReg": FittedPricesReg, "FittedDeltaReg": FittedDeltasReg}
df = pd.DataFrame(data)
dfsort = df.sort_values("S0")
plt.subplot(1, 2, 1)
plt.scatter(S0, DiscPayoff, color = "lightgrey", label = "Simulated Discounted Payoffs by price-only-regression")
plt.scatter(S0, DiscPayoffReg, color = "grey", label = "Simulated Discounted Payoffs by Delta-regularization")
plt.plot(dfsort["S0"], dfsort["FittedPrice"], label = "Fitted Price by price-only-regression", color ="red")
plt.plot(dfsort["S0"], dfsort["FittedPriceReg"], label = "Fitted Price by Delta-regularization", color ="blue")
plt.plot(dfsort["S0"], dfsort["BinPrice"], label = "Binomial American price with " + str(TimeStep) + " time-steps", color = "black")
plt.xlabel('Time-zero Stock-price')
plt.ylabel('Time-zero Amr-put Price')
plt.title('American Put option with K = ' + str(K))
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(S0, D, color = "grey", label = "Simulated D-variable by Delta-regularization")
plt.plot(dfsort["S0"], dfsort["FittedDelta"], label = "Fitted Delta by price-only-regression", color ="red")
plt.plot(dfsort["S0"], dfsort["FittedDeltaReg"], label = "Fitted Delta by Delta-regularization", color ="blue")
plt.plot(dfsort["S0"], dfsort["BinDelta"], label =  "Binomial American Delta with " + str(TimeStep) + " time-steps", color = "black")
plt.xlabel('Time-zero Stock-price')
plt.ylabel('Time-zero Amr-put Delta')
plt.title('American Put option with K = ' + str(K))
plt.legend()
plt.show()


