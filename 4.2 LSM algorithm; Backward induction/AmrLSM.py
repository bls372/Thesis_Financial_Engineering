import numpy as np
S0 = 1
K = 1
r = 0.06
sigma = 0.2
capT = 1
# This functions gives an approximate of a at-the-money American price
# using LSM introduced by Longstaff&Schwartz by backward induction
# as a function of one-dimensional stock price, time to maturity (capT),
# number of exercisable time-points (Nhedge), number of simulations (NoSim),
# and number of degree in the polynomial fit (p) for approximating the
# continuation-values
def AmrLSM(spot, capT, Nhedge, NoSim, p):
    S0 = spot
    n = NoSim
    dt = capT/Nhedge
    St = np.zeros((n, Nhedge +1))
    St[:,0] = S0
    #The stock-price paths are first simulated all the way to expiry
    for i in range(Nhedge):
        Z = np.random.normal(0, 1, size=n)
        St[:,i+1] = St[:,i]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    #Then the algorithm runs as it has been explained 
    PayOffMatrix = (K-St)*(St<K)
    DiscCashflow = np.exp(-r*dt)*PayOffMatrix[:,Nhedge]
    StopRule = np.zeros((n, Nhedge+1))
    StopRule[:,Nhedge] = 1 
    for i in range(Nhedge-1):
        q = Nhedge -1 -i
        inMonPosition = np.where(PayOffMatrix[:,q] !=0)[0]
        x = St[inMonPosition, q]
        P = DiscCashflow[inMonPosition]
        X = np.vander(x-K, p+1, increasing=True)
        theta = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X))
                          , np.matmul(np.transpose(X), P))
        contVal = np.matmul(X, theta)
        exVal = PayOffMatrix[inMonPosition,q]
        DiscCashflow[inMonPosition]=exVal*(exVal>=contVal)*1*(exVal>0)+P*(exVal<contVal)
        StopRule[inMonPosition, q] = (exVal>=contVal)*1*(exVal>0)
        DiscCashflow = np.exp(-r*dt)*DiscCashflow
        for j in range(2, Nhedge+1):
            StopRule[:,range(j, Nhedge+1)] = (StopRule[:,range(j, Nhedge+1)].T
                                              *(StopRule[:,j-1] == 0)).T
    Price0 = np.mean(DiscCashflow)
    Price0Prime = np.exp(-r*dt*np.array(range(Nhedge+1)))*PayOffMatrix*StopRule
    Price0PrimePrime = Price0Prime.sum(axis=1)
    Price02 = np.mean(Price0PrimePrime)
    result = {"Price0": Price0, "Price02": Price02, "StopRule": StopRule}
    return(result)
print(AmrLSM(spot=S0, capT = 1, Nhedge = 52, NoSim = 100000, p=3))
