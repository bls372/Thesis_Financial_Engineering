import numpy as np
S0=1
K = 1
r = 0.06
sigma = 0.2
p1 = 3
#This the extended version of LSM using Backward-induction presented earlier
#This function is first run before the forward-pass is run
def AmrLSM(spot, capT, Nhedge, NoSim, p, method):
    S0 = spot
    n = NoSim
    dt = capT/Nhedge
    w = 0.5
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
#Given the coefficients from Backward-induction, this function gives an approximate
# of a at-the-money American put option going forward in the stock-price simulation
def AmrLSMforward(spot, capT, Nhedge, NoSim, p, coefficients):
    S0 = spot
    n = NoSim
    dt = capT/Nhedge
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
M = AmrLSM(spot=S0, capT=1, Nhedge=52, NoSim=10000, p=p1, method="Delta-regularization")
coeffs = M["Coefficients"]
print(AmrLSMforward(spot=S0, capT=1, Nhedge=52, NoSim=10000, p=p1, coefficients=coeffs))

