import numpy as np
S0=1
K = 1
sigma = 0.2
r = 0.06
capT = 1
#This is a function that calculates the Binomial price and Delta of
# an European Put option as a function of the current spot price,
# time to maturity, and number of time-points in the lattice
def EuPutBino(spot, TimeToMat, NoTimePoints):
    S0 = spot
    dt = TimeToMat/NoTimePoints
    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))
    R = np.exp(r*dt)
    q = (R-d)/(u-d)
    St = np.ones((NoTimePoints+1,NoTimePoints+1))
    St[0][0] = S0
    # Creating the stock-price lattice, which is really irrelevant, as
    # it is only the terminal stock-prices that is needed, and these can be
    # calculated by using Binomial formula
    for k in range(NoTimePoints+1):
        if k>0:
            St[k][k] = d*St[k-1][k-1]
        if k<(NoTimePoints):
            for j in range(k,NoTimePoints):
                St[k][j+1] = u*St[k][j]
    payoff = (K-St)*(St<K)
    cashflow = np.zeros((NoTimePoints+1,NoTimePoints+1))
    cashflow[:,NoTimePoints] = payoff[:,NoTimePoints]
    #Starting at expiry, the price is recursively calculated in each node
    # until time zero is reached.
    for i in range(NoTimePoints):
        l = NoTimePoints-i
        for j in range(l):
            cashflow[j][l-1] = (1/R)*(q*cashflow[j][l] + (1-q)*cashflow[j+1][l])
    # The time zero Delta is the sensitivity of the Price
    # with resprect to the stock-price change
    Delta = (cashflow[0][1]-cashflow[1][1])/(St[0][1]-St[1][1])
    result = {"Price": cashflow[0][0], "Delta": Delta}
    return(result)
print(EuPutBino(spot = S0, TimeToMat = capT, NoTimePoints = 750))
