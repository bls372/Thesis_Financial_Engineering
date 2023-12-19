import numpy as np
S0=1
K = 1
sigma = 0.2
r = 0.06
capT = 1
# This function gives the time zero price and Delta by the stnadard
# Binomial model as a function of the current stock price, time to maturity
# and number of time points in the lattice.
def AmrPutBino(spot, TimeToMat, NoTimePoints):
    S0 = spot
    dt = TimeToMat/NoTimePoints
    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))
    R = np.exp(r*dt)
    q = (R-d)/(u-d)
    St = np.ones((NoTimePoints+1,NoTimePoints+1))
    St[0][0] = S0
    #In contrast to European options, one now need the whole lattice
    # for pricing American options
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
            #These lines are only that are different from EuPutBino, which follows
            # exactly the construction of the Snell-Envelope
            exVal = payoff[j][l-1]
            contVal = (1/R)*(q*cashflow[j][l] + (1-q)*cashflow[j+1][l])
            cashflow[j][l-1] = exVal*(exVal>= contVal) + contVal*(exVal<contVal)
    # The time zero Delta is still the senitivity in price change with respect
    # to the sensitity in stock price change
    Delta = (cashflow[0][1]-cashflow[1][1])/(St[0][1]-St[1][1])
    result = {"Price": cashflow[0][0], "Delta": Delta}
    return(result)
print(AmrPutBino(spot=S0, TimeToMat=capT, NoTimePoints=750))
